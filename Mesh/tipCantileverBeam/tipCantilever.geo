// Gmsh project created on Sun Jun 27 19:24:03 2021
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 20, 0};
//+
Point(3) = {40, 20, 0};
//+
Point(4) = {40, 3, 0};
//+
Point(5) = {40, 0, 0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 1};
//+
Curve Loop(1) = {2, 3, 4, 5, 1};
//+
Plane Surface(1) = {1};
//+
Physical Curve("fixedX", 6) = {1};
//+
Physical Curve("fixedY", 7) = {1};
//+
Physical Curve("forceX", 8) = {4};
//+
Physical Curve("forceY", 9) = {4};
//+
Physical Curve("insulated", 10) = {2, 5, 3};
//+
Physical Surface("body", 11) = {1};
