
import cadquery as cq

from cadquery.vis import show

rows = 8
cols = 8
pitch = 8

dx = cols*pitch
dy = rows*pitch
dz = 3*pitch

#roic
assy = cq.Workplane('XY').center(0,0).box(dx,dy,dz, centered=True)

#bumps
r = pitch/2/2
h = (2*r)*2
assy = assy.faces('>Z').workplane(offset=h/2).rarray(pitch,pitch,cols,rows).cylinder(h,r)

cq.exporters.export(assy, "assy-1.step", exportType = "STEP")

show(assy)

