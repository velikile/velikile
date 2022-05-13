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
