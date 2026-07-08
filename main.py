import depthai as dai
import time
import sys
import numpy as np
import cv2

try:
    import open3d as o3d
except ImportError:
    sys.exit("cant find open3d")

isRunning = True

def draw_crosshair(heatmap, depth_frame, x, y, color=(255, 255, 255), size=10, thickness=1):
    z = depth_at(depth_frame, x, y)
    cv2.drawMarker(heatmap, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness)
    if z:
        cv2.putText(heatmap, f"{z/1000:.2f} m", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return heatmap

def depth_at(depth_frame, x, y):
    if x < 0 or y < 0 or x >= depth_frame.shape[1] or y >= depth_frame.shape[0]:
        return None
    roi = depth_frame[max(0, y-2):y+3, max(0, x-2):x+3]
    roi = roi[roi > 0]
    if roi.size == 0:
        return None
    return np.median(roi)

class O3DNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.inputPCL = self.createInput()

    def run(self):
        def key_callback(vis, action, mods):
            global isRunning
            if action == 0:
                isRunning = False
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="PointCloud", width=640, height=480)
        vis.register_key_callback(ord("Q"), key_callback)
        pcd = o3d.geometry.PointCloud()
        coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
        vis.add_geometry(coordinateFrame)
        first=True

        while isRunning and self.isRunning():
            try:
                inPCL = self.inputPCL.tryGet()
            except dai.MessageQueue.QueueException:
                return

            if inPCL is not None:
                points, colors = inPCL.getPointsRGB()
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                colors = (colors / 255.0).astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(np.delete(colors, 3, 1))

                if first:
                    vis.add_geometry(pcd)
                    first=False
                else:
                    vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

        vis.destroy_window()

# retry logic
pipeline = None
for attempt in range(4):
    try:
        pipeline = dai.Pipeline()
        break
    except Exception as e:
        print(f"connect attempt {attempt + 1} failed: {e}")
        time.sleep(10)
if pipeline is None:
    sys.exit("could not connect to the device after 4 attempts")

with pipeline:
    fps = 17
    resolution = (640, 400)

    left = pipeline.create(dai.node.Camera)
    right = pipeline.create(dai.node.Camera)
    stereo = pipeline.create(dai.node.StereoDepth)
    color = pipeline.create(dai.node.Camera)
    rgbd = pipeline.create(dai.node.RGBD).build()
    align = None
    color.build()
    o3dViewer = pipeline.create(O3DNode)
    left.build(dai.CameraBoardSocket.CAM_B)
    right.build(dai.CameraBoardSocket.CAM_C)
    out = None

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setRectifyEdgeFillColor(0)
    stereo.enableDistortionCorrection(True)

    left.requestOutput(resolution).link(stereo.left)
    right.requestOutput(resolution).link(stereo.right)
    platform = pipeline.getDefaultDevice().getPlatform()
    if platform == dai.Platform.RVC4:
        out = color.requestOutput(resolution, dai.ImgFrame.Type.RGB888i, enableUndistortion=True)
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        out.link(align.inputAlignTo)
        depthOut = align.outputAligned
        align.outputAligned.link(rgbd.inDepth)
    else:
        out = color.requestOutput(resolution, dai.ImgFrame.Type.RGB888i, dai.ImgResizeMode.CROP, fps, True)
        stereo.depth.link(rgbd.inDepth)
        depthOut = stereo.depth
        out.link(stereo.inputAlignTo)

    depthQ = depthOut.createOutputQueue(maxSize=4, blocking=False)
    rgbQ = out.createOutputQueue(maxSize=4, blocking=False)

    out.link(rgbd.inColor)
    rgbd.pcl.link(o3dViewer.inputPCL)

    pipeline.start()
    print("pipeline running - press Q in a window to quit")

    mouse = {"pos": None}
    def on_mouse(event, x, y, flags, param):
        mouse["pos"] = (x, y)

    cv2.namedWindow("Depth Heatmap")
    cv2.setMouseCallback("Depth Heatmap", on_mouse)

    while isRunning and pipeline.isRunning():
        inRgb = rgbQ.tryGet()
        if inRgb is not None:
            cv2.imshow("RGB", inRgb.getCvFrame())

        inDepth = depthQ.tryGet()
        if inDepth is not None:
            depthFrame = inDepth.getFrame()
            clipped = np.clip(depthFrame, 0, 10000)
            norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_TURBO)
            heatmap[depthFrame==0] = (30,30,30)

            h, w = depthFrame.shape
            draw_crosshair(heatmap, depthFrame, w // 2, h // 2)
            if mouse["pos"]:
                mx, my = mouse["pos"]
                draw_crosshair(heatmap, depthFrame, mx, my, color=(0, 0, 255))

            cv2.imshow("Depth Heatmap", heatmap)

        if cv2.waitKey(1) in (ord('q'), ord('Q')):
            isRunning = False

    cv2.destroyAllWindows()



    


