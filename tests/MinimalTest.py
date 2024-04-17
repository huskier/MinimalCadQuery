import minimalcadquery as cq
import sys
sys.dont_write_bytecode = True

rectBasedObject = cq.Workplane().rect(10, 20).extrude(50)

cq.exporters.export(rectBasedObject, "rectBasedObject.step", exportType = "STEP")

box = cq.Workplane().box(50, 50, 50)

cq.exporters.export(box, "box.step", exportType = "STEP")
