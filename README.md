# grid-cells

---
# ATTENTION!

After a discussion with Vemund, Konstantin has changed the code methods.py 
such that the generation of grid cells is more intuitive. Instead of specifying
the spatial frequency of the generating planar waves $f$, the user is supposed to
specify the spatial period (lattice constant) **a** of the grid cell lattice in 
the maxima of activity itself, since **f is not a**. 

Unfortunately, this may result in your code breaking. So please change your code
accordingly or ask Vemund about that. 

If you guys find it more intuitive to use the spatial frequency instead of the
lattice constant, you are invited to change the code respectively by subsidising
$a$ for $a / f$. 

If there are any questions about the changes, you can ask any of us or read the
Grid-Theory notbook (see link below).

---
## Grid Theory

A comprehensive summary of the lattice-theoretic basics that are necessary to 
understand what we did in this project and how they relate to the 
grid cell ratemap pattern and its interpretation as a planar hexgonal lattice
is provided in the 
[Grid-Theory notebook](https://github.com/Vemundss/grid-cells/blob/main/Grid-Theory.ipynb).

---
