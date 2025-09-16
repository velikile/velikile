Credit [Bezier curve](https://en.wikipedia.org/wiki/B%C3%A9zier_curve)
```
def point_on_curve(p0,p1,p2,p3,t)
    return t(t(p0) + (1-t)(p1)) + (1-t)(t(p2) + (1-t)(p3))

```
![Bézier_3_big](https://user-images.githubusercontent.com/7438866/171404406-e1621cf1-89f8-4d27-9f8f-f9c6b91d132f.gif)
![Bézier_4_big](https://user-images.githubusercontent.com/7438866/171403914-468bfc7c-ac86-46bc-b5f5-47b3b9dd9b87.gif)


Credit: [Barycentric Coordinates](https://www.cut-the-knot.org/triangle/barycenter.shtml).

Given any three points in R3 {A,B,C} a point p on the plane enclosed by ABC can be defined by three weights w0 , w1 ,w2 
such that

w0 + w1 + w2 = 1 

p = A * w0 + B * w1 + C * w2 -> p is no the surface of the triangle 

![image](https://user-images.githubusercontent.com/7438866/170110522-b46b5606-1071-4a86-ae53-2293fa0b42bf.png)
```

def point_in_triangle(v0,v1,v2 ,N , P):
    edge0 = v1 - v0 
    edge1 = v2 - v1 
    edge2 = v0 - v2
    C0 = P - v0 
    C1 = P - v1 
    C2 = P - v2

    if dotProduct(N, crossProduct(edge0, C0)) > 0 and dotProduct(N, crossProduct(edge1, C1)) > 0 and dotProduct(N, crossProduct(edge2, C2)) > 0 :    
        return True
    return False
    
```


Credit: [Computational Geometry](https://www.toptal.com/python/computational-geometry-in-python-from-theory-to-implementation).
```
# python line segment intersection test


def ccw(A, B, C):
    """Tests whether the turn formed by A, B, and C is ccw 
    return (B.x - A.x) * (C.y - A.y) > (B.y - A.y) * (C.x - A.x)
    
def intersect(a1, b1, a2, b2):
    """Returns True if line segments a1b1 and a2b2 intersect."""
    return ccw(a1, b1, a2) != ccw(a1, b1, b2) and ccw(a2, b2, a1) != ccw(a2, b2, b1)
    
```

Credit [Ray Tracing in a weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html#rays,asimplecamera,andbackground).

```
double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = half_b*half_b - a*c;

    if (discriminant < 0) {
        return -1.0;
    } else {
        return (-half_b - sqrt(discriminant) ) / a;
cds                    }
}
```

Credit [Solving the Right Problems for Engine Programmers - Mike Acton‌ (TGC 2017)](https://www.youtube.com/watch?v=4B00hV3wmMY&t=87s)
#### Reasonable defaults.
- Linear search through array.
- FIFO managed by integer.
- Store objects by types explicitly.
- Batch processing as a desgin goal since it's the most common case.
- Excplicit latency and throughput constraints. when can I issue the next instruction.
- Version data formats.
- Allocators block ,stack ,scratch.
- Model target manually first visualize the target result somehow.
- Index look aside table.

Credit : [Andrew Kelley Practical Data Oriented Design] (DoD)(https://www.youtube.com/watch?v=IroPQ150F6c)

#### Main points 
- Reduce size of the structs by using indices instead of pointers
- For booleans use Arrays for entities (dead\ alive) in seperate arrays "store them out of band"
- Make sure the order of elements doesn't conflict with the allignment requirements
- Use SOA instead of AOS to reduce padding zig has multiarrayilst converts structs into SOA lists automagically
- Use observations about the behaviour of the code , if you see waste you can store it out of band in a "sparse array(hash map)"
- Reduce size of structs by using the Encoding technique group the common data and store the uncommon data in an out of band manner


Credit [Dennis Gustafsson – Parallelizing the physics solver – BSC 2025](https://www.youtube.com/watch?v=Kvsvd67XUKw&ab_channel=BetterSoftwareConference)
#### Main ideas 
- Projected gauss seidel Sequential impulse (sequentially satisfy constrained and repeat until solved)
- [Graph Coloring (Box 2d Erin Cotto)](https://box2d.org/posts/2024/08/simd-matters/)
-     Detailed rigid body simulation (Nvidia R&D) https://youtu.be/zzy6u1z_l9A
- Use Parallel_for for simplicity sake
- condition variables are slow for quick wake up (busy wait loops are faster for quick wakeup)
- Use profiler or include a way to visualize the timings information in the program

Credit [Sean Barrett stbcc ](https://stb.handmade.network/blog/p/1136-connected_components_algorithm)
#### Main pooints
for the connected component part use the [disjoint set forests](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)
```
struct point
{
    uint16_t x;
    uint16_t y;
};
point leader[1024][1024];

point dsf_find(int x, int y)
{
   point p,q;
   p = leader[y][x];
   if (p.x == x && p.y == y)
      return p;
   q = dsf_find(p.x, p.y);
   leader[y][x] = q; // path compression
   return q;
}

void dsf_union(int x1, int y1, int x2, int y2)
{
   point p = dsf_find(x1,y1);
   point q = dsf_find(x2,y2);

   if (p.x == q.x && p.y == q.y)
      return;

   leader[p.y][p.x] = q;
}

```
#### Deletion problem
The connected components data structure contains a label for each node , that label is the same one for the nodes that has a connection to any of the other nodes in that tree . When trying to delete a connection a problem arises the compnents is now split , the worst case is on the delete you would have to update half of the nodes . An option to tackle that problem is to store the label into a different location and only store a pointer to that location inside the nodes that lie within that components , so that after the update you only update the label. But when you disconnect a large component into components we might make that disconnection in many places , there's no way to predict the layout of the pointed to labels such that it matches our disconnection in a way where we only need to update k labels. Most pepole just rebuild the entire connected components data structre (recompute the flood fill every nth frame). The idea of stbcc is to compromise introduce a moderate amount of intermediate indirected labels so that we can rewrite as few labels as possible and minimize the number of nodes which we need to point to different labels entirely , we do this by utilizing the assumption that we have a grid and the geometry of it can guide us.


#### Two level hierarchies 
There is a way to make a n^2 algorithm into a n^1.5 algorithm 
by splitting the data points into a grid of size sqrt(n) * sqrt(n) run the algorithm on the new pieces , each of these peices will take O(sqrt(n)) so the entire set will take O(sqrt(n) * n) which is O(n^1.5)

Two levels heirarchies are almost always better than quadtrees or other fully recursive algorithms in terms of speeding things up. 

#### On every update
1. rebuild the locally connected components for the subgrid
2. rebuild the adjacency from that subgrid to its neighbors and vice versa
3. compute from scratch the globally connected components forom the locally connected pieces

Credit [PCG, a family of better random generators](https://www.pcg-random.org/#)
1. linear congruential generator (LCG) sequential psaudo random generator
   $X_{n+1} = (aX_n + c) mod  {m}$
   That produces a repeating pattern which is not random enough to be used for say a ray tracing
2. Use the lcg output as the input to a bit mixer 
 ```
    state = state * m + i ;
    output1 = state >> (29 - (state >> 61))
    output2 = rotate32((state ^ state>>18) >> 27 , state >> 59)
    //just an example with made up numbers
    output3 = state ^ (state >> 11 ^ state <<7 ^ state >> 3) 
```
