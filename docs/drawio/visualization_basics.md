## Data Arrangement

In the case of PRISMA, a cube has dimensions arranged in a very specific manner. Out of the box, each cube has the dimensions

$$
H \times C \times Width
$$

Here, oddly the channels are in the middle. 

Dimension 0 is height also called the Along track dimension which is N-S. Dimension 2 is the cross track dimension which is E-W. Dimension 1 is the channel dimension. This is called the Band Interleaved by Line (BIL) format. Python for visualization expects a different format called the Band Interleaved by Pixel (BIP) format.Therefore, before visualization this has to be transformed into a 

$$
H \times W \times C
$$

## Handling Transformations

Moving between these formats will be crucial for visualization and training. Look up the section on image transformation to see how this is done.