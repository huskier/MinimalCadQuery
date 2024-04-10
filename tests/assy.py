
import minimalcadquery as cq

from minimalcadquery.vis import show

rows = 5
cols = 5
pitch = 8

dx = cols*pitch
dy = rows*pitch
dz = 3*pitch
roic = cq.Workplane('XY').box(dx,dy,dz, centered=True)

show(roic)

r = pitch/2/2
h = (2*r)*2
bumps = cq.Workplane('XY').rarray(pitch,pitch,cols,rows, center=True).cylinder(h,r)

#bumpsBoundingBox = bumps.val().
#print(type(bumps.val()))

face_num = bumps.faces("<Z").size()
print("face number is ", face_num)

for i in range(25):
    print(bumps.faces("<Z").vals()[i].Center().x)
    print(bumps.faces("<Z").vals()[i].Center().y)
    #print(bumps.faces("<Z").vals()[i].Center.z())    
    #show(bumps.faces("<Z").vals()[i])

assy = cq.Assembly()

assy.add(roic, name='roic', color=cq.Color('pink'))
assy.add(bumps, name='bumps', color=cq.Color('gray'))
assy.constrain("roic@faces@>Z", "bumps@faces@<Z", "Plane")

'''
assy.add(roic, name='roic', color=cq.Color('pink'))
assy.add(bumps, name='bumps', color=cq.Color('gray'))
assy.constrain("roic@faces@>Z", "bumps@faces@<Z", "Axis")
assy.constrain("roic@faces@>Z", "bumps@faces@<Z", "PointInPlane")
#assy.constrain("roic@faces@<Y", "bumps@faces@>Y", "Axis")
assy.constrain("roic", "FixedRotation", (0, 0, 0))
assy.constrain("bumps", "FixedRotation", (0, 0, 0))
assy.constrain("roic@faces@>Z", "bumps", "Point", param=(h/2))
'''

assy.solve()
#display(assy)
show(assy)