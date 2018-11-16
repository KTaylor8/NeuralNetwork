import matplotlib.pyplot as plt
import math
import pylab
import numpy

x = numpy.linspace(-20,20,100)
y = numpy.sin(x)

plt.plot(x,y)
plt.axis([-20,20,-5,5])
plt.show()