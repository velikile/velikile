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
    C2 = P - v2![Bézier_3_big](https://user-images.githubusercontent.com/7438866/171404270-dfcc24a3-27c9-4f15-be2e-81f4e31e292f.gif)

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
    }
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

