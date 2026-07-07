import depthai as dai
from datetime import datetime, time, timedelta
import sys
import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("cant find open3d")

class O3DNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().init()
        dai.node.ThreadedHostNode.__init__(self)
        self.inputPCL = self.createInput()

        def run(self):
            def key_callback(vis,action, mods):
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

            while self.mainLoop():
                try:
                    inPCL = self.inputPCL.tryGet()
                except dai.NessageQueue.QueueException:
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

# create the pipeline
with dai.Pipeline() as pipeline:
    fps = 30

    left = pipeline.create(dai.node.Camera)
    stereo = pipeline.create(dai.nodeStereoDepth)
    color = pipeline.create(dai.node.Camera)
    rgbd = pipeline.create(dai.node.RGBD).build()
    align = None
    color.build()
    o3dViewer = pipeline.create(O3DNode)
    left.build(dai.CameraBoardSocket.CAM_B)
    out = None

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setRectifyEdgeFillColor(0)
stereo.enableDistortionCorrection(True)

left.requestOutput((640, 400)).link(stereo.left)
platform = pipeline.getDefaultDevice().getPlatform()
if platform == dai.Platform.RVC4:
    out = color.requestOutput((640, 400), dai.ImgFrame.Ztype.RGB888i, enableUndistortion=True)
    align = pipeline.create(dai.node.ImageAlign)
    stereo.depth.link(align.input)
    out.link(align.inputAlignTo)
    align.outputAligned.link(rgbd.inDepth)
else:
    out = color.requestOutput((640, 400), dai.ImgFrame.Ztype.RGB888i, dai.ImageResizeMode.CROP, 30, True)
    stereo.depth.link(rgbd.inDepth)
    out.link(stereo.inputAlignTo)

out.link(rgbd.inColor)
rgbd.pcl.link(o3dViewer.inputPCL)

pipeline.start()
while pipeline.isRunning():
    time.sleep(1)



    


