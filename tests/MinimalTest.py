import minimalcadquery as cq

rectBasedObject = cq.Workplane().rect(10, 20).extrude(50)

cq.exporters.export(rectBasedObject, "rectBasedObject.step", exportType = "STEP")