import minimalcadquery as cq

from minimalcadquery.vis import show

cone = cq.Solid.makeCone(1, 0, 2)

assy = cq.Assembly()
assy.add(cone, name="cone0", color=cq.Color("green"))
#show(assy)
assy.add(cone, name="cone1", color=cq.Color("blue"))
#show(assy)
assy.constrain("cone0@faces@<Z", "cone1@faces@<Z", "Axis")

assy.solve()

assy.save('assy.step')

#show(assy)