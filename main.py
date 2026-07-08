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

def draw_crosshair(depth_frame, x, y, color=(0, 0, 255), size=5, thickness=1):
    h, w = depth_frame.shape
    cx, cy = w // 2, h // 2
    z = depth_at(depth_frame, cx, cy)
    cv2.drawMarker(depth_frame, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness)
    if z:
        cv2.putText(depth_frame, f"{z:.1f} mm", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return depth_frame

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

    left.requestOutput((640, 400)).link(stereo.left)
    right.requestOutput((640, 400)).link(stereo.right)
    platform = pipeline.getDefaultDevice().getPlatform()
    if platform == dai.Platform.RVC4:
        out = color.requestOutput((640, 400), dai.ImgFrame.Type.RGB888i, enableUndistortion=True)
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        out.link(align.inputAlignTo)
        depthOut = align.outputAligned
        align.outputAligned.link(rgbd.inDepth)
    else:
        out = color.requestOutput((640, 400), dai.ImgFrame.Type.RGB888i, dai.ImgResizeMode.CROP, 30, True)
        stereo.depth.link(rgbd.inDepth)
        depthOut = stereo.depth
        out.link(stereo.inputAlignTo)

    depthQ = depthOut.createOutputQueue(maxSize=4, blocking=False)
    rgbQ = out.createOutputQueue(maxSize=4, blocking=False)

    out.link(rgbd.inColor)
    rgbd.pcl.link(o3dViewer.inputPCL)

    pipeline.start()
    print("pipeline running - press Q in the viewer window to quit")
    while isRunning and pipeline.isRunning():
        inDepth = depthQ.tryGet()
        if inDepth is not None:
            depthFrame = inDepth.getFrame()
            clipped = np.clip(depthFrame, 0, 10000)
            norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_TURBO)
            heatmap[depthFrame==0] = (30,30,30)
        time.sleep(0.1)



    


