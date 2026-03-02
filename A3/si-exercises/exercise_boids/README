# (Partial) Boids implementation

In this folder, you will find a (partial) implementation of Reynold's Boids.

# Step 1: check out the HTML implementation

First open boids.html in your browser. You should see a field with boids, 
but the boids do not yet move. That is because the update rule still needs to 
be implemented. 

Open the file "boids.js" in a text editor. You will find a configuration object 
(starting in line 5), where you can update the standard parameter values. 
The Canvas class (starting line 14) takes care of drawing the field; you don't need 
to update it unless you want to. The Scene class (starting line 83) contains stuff
regarding the space where the swarm lives, and takes care of positioning, neighborhoods,
etc. Most important will be the Particle class (starting line 220). Here, you will 
need to implement the alignmentVector, cohesionVector, and separationVector methods
that evaluate the corresponding Boid rules within a given radius of the current boid of 
interest. You will then need to complete the updateVector method to bring all components
together correctly and update the position and direction of the current Boid 
accordingly.

# Step 2: Fix the HTML implementation

In boids.js, make the required updates to implement the Boid update rule by completing
the Particle class. Verify that if you open boids.html in your browser, your implementation
works. You should be able to adjust parameters with the input sliders.

# Step 3: Run code in Node.js to get quantitative output and/or images

Make sure you have Node.js and its package manager npm installed. 
Check out these instructions: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm

Then, from your command line, run:

```
npm install
```

To install required packages (you won't need many, they are listed in package.json).

Verify that you can now run the node script:

```
node node-boids-template.js
```

It will write quantitative output to the console in csv format, with columns for
simulation time step, boid ID, x and y coordinates. You can directly write this
output to a csv if you want:

```
node node-boids-template.js > myOutput.csv
```

Then, you can import the data for analysis however you prefer (e.g. in python or R).

You can also output PNG images:

```
node node-boids-template.js -I output-img
```

This will create a folder output-img with the images from this simulation. 


# Step 3: Run code in Node.js to get quantitative output and/or images

The provided template does not work yet because, just like the HTML before,
the Particle class has not yet been completed. 

Copy your Particle class from boids.js into the node script (replace the one that is 
currently there).

If you now run again: 

```
node node-boids-template.js > myOutput.csv
```

You should see that the coordinates actually change from step to step. 

# Step 4 : Generate outputs

The script has some command line argument that allows you to run simulations at 
different parameters or with different outputs.

For example, you can run:

```
node node-boids-template.js -f 500 -N 50 -i 10 -o 25 -c 1 -s 1 -a 1 -T 5000
```

To run a simulation on a 500 x 500 field for T = 5000 steps, 
with N = 50 boids, an inner (i) and outer (o)
radius of 10 and 25, respectively, and with cohesion (c), separation (s), and alignment (a)
weights all 1. 


