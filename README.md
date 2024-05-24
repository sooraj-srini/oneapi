# Introduction

I don't have a nvidia gpu card, so I am forced to use Iris XE graphics. While I do have the intel gpu card, I might as well use it to my fullest extent, you know what I mean?

This is me learning oneapi. I still prefer cuda tho rn.

# Assignment 3 (may 24 2024)

This (a3.cpp) is my SyCL implementation of CS6023: GPU Programming Assignment 3. The assignment is a scene renderer with multiples meshes and stuff.

More importantly, the code works and is very performant (as is to be expected from a GPU implementation) being roughly the same as the CUDA implementation. 

Doubts I have with sycl still:
- is the way i used atomic_ref correct? 
- i have no idea how synchronization works or whether there are warps in sycl
- memory management is kinda weird but i trust sycl now at least
- i hate the fact that i can't use two parallel_for in one kernel?? what is the point then? I can't even reuse the ever growing accessors??
- its not that bad.