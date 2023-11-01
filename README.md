# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

swf is an improved 3D mesh model simplification program based on triangle folding algorithm.

### How do I get set up? ###

The Pycharm compiler is recommended,other python compilers are also available, and the following dependencies need to be satisfied:
1) Use python3.0 or later

2) Install the following python packages: 
- numpy
- scipy
- matplotlib
- tensorflow

If you do not have the above packages, you can install it as follows:
pip install numpy
pip install scipy
pip install matplotlib
pip install tensorflow

3) getdata.py    Get information about the original file(vertex coordinates, face index)
calculate.py     Judge the vertex type, calculate the vertex normal vector and triangle area
foldPrice.py     The Mahalanobis distance of the five constraint factors is calculated, and then the folding cost is calculated
new_cal.py       Update the information of each triangular face
new_fold.py      Update the folding cost of each triangle
start_zd.py      At the entrance of the whole program, the final simplified result is obtained and the line chart is used to evaluate the result
model1.obj,model2.obj,model3.obj      Original 3D building mesh model 

Finally, the simplified model can be visualized using the open source software MeshLab
### Who do I talk to? ###

Shen Wenfei, Beijing University of Civil Engineering and Architecture