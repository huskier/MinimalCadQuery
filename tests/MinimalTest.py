import minimalcadquery as cq
import logging

logging.basicConfig(filename='test_logging_1.txt',format='[%(asctime)s,%(msecs)03d-%(filename)s-line_%(lineno)d-%(levelname)s:%(message)s]', level = logging.DEBUG,filemode='w',datefmt='%Y-%m-%d%I:%M:%S %p')
logger = logging.getLogger("minimalcadquery")


if __name__=='__main__':
    
    logger.info("In MinimalTest.py file......")

    rectBasedObject = cq.Workplane().rect(10, 20).extrude(50)
    cq.exporters.export(rectBasedObject, "rectBasedObject.step", exportType = "STEP")

    box = cq.Workplane().box(50, 50, 50)
    cq.exporters.export(box, "box.step", exportType = "STEP")
