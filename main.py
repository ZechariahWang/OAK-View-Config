import depthai as dai
import time
import sys
import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("cant find open3d")

isRunning = True

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
    fps = 30

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
        align.outputAligned.link(rgbd.inDepth)
    else:
        out = color.requestOutput((640, 400), dai.ImgFrame.Type.RGB888i, dai.ImgResizeMode.CROP, 30, True)
        stereo.depth.link(rgbd.inDepth)
        out.link(stereo.inputAlignTo)

    out.link(rgbd.inColor)
    rgbd.pcl.link(o3dViewer.inputPCL)

    pipeline.start()
    print("pipeline running - press Q in the viewer window to quit")
    while isRunning and pipeline.isRunning():
        time.sleep(0.1)



    


