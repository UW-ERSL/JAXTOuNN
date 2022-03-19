// Gmsh project created on Sun Jun 27 20:15:37 2021
//+
Point(1) = {0, 0, 0};
//+
Point(2) = {60, 0, 0, 1};
//+
Point(3) = {60, 13, 0, 1};
//+
Point(4) = {60, 17, 0, 1};
//+
Point(5) = {60, 30, 0, 1};
//+
Point(6) = {0, 30, 0, 1};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6,1};
//+
Curve Loop(1) = {2, 3, 4, 5, 6, 1};
//+
Plane Surface(1) = {1};
//+
Physical Curve("fixedX", 6) = {5};
//+
Physical Curve("fixedY", 7) = {5};
//+
Physical Curve("forceX", 8) = {3};
//+
Physical Curve("forceY", 9) = {3};
//+
Physical Curve("insulated", 10) = {2, 3, 4, 1};
//+
Physical Surface("body", 11) = {1};
//+
Transfinite Surface {1} = {6, 5, 2, 1};
//+
Transfinite Curve {6, 4, 3, 2} = 10 Using Progression 1;
//+
Transfinite Curve {5, 1} = 10 Using Progression 1;
//+
Transfinite Curve {4, 2} = 4 Using Progression 1;
//+
Transfinite Curve {3} = 2 Using Progression 1;
//+
Transfinite Curve {3} = 2 Using Progression 1;
//+
Transfinite Curve {4, 2} = 4 Using Progression 1;
//+
Transfinite Curve {6, 1, 5} = 10 Using Progression 1;
