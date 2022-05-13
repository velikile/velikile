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

