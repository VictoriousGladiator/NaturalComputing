const args = require('minimist')(process.argv.slice(2));
const fs = require( 'fs' )

/* =========  copy your Particle class here:   */



class Particle {
	constructor( Scene, i ){
		this.S = Scene
		this.speed = 1 
		this.id = i
		this.pos = this.S.randomPosition() 
		this.dir = this.S.randomDirection()
	}
	// return a + b
	addVectors( a, b ){
		const dim = a.length
		let out = []
		for( let d = 0; d < dim; d++ ){
			out.push( a[d] + b[d] )
		}
		return out
	}
	// return a - b
	subtractVectors( a, b ){
		const dim = a.length
		let out = []
		for( let d = 0; d < dim; d++ ){
			out.push( a[d] - b[d] )
		}
		return out
	}
	// multiply vector by a constant
	multiplyVector( a, c ){
		return a.map(( x ) => x * c ) 
	}
	// normalize vector to unit length
	normalizeVector( a ){
		return this.S.normalizeVector(a)
	}
	
	// should return a unit vector in average neighbor direction for neighbors within 
	// distance neighborRadius 
	alignmentVector( neighborRadius ){
		
		return this.dir
	
	}
	
	// should return a unit vector in the direction from current position to the 
	// average position of neighbors within distance neighborRadius 
	cohesionVector( neighborRadius ){
		
		return this.dir
	
	}
	
	// as cohesionVector, but now return the opposite direction for the given 
	separationVector( neighborRadius ){
		
		return this.dir 
		
	}
	
	updateVector(){
		
		let align_weight = this.S.conf.alignment
		let cohesion_weight = this.S.conf.cohesion
		let separation_weight = this.S.conf.separation
		
		const align = this.multiplyVector( this.alignmentVector( this.S.conf.outerRadius ), align_weight )
		const cohesion = this.multiplyVector(this.cohesionVector( this.S.conf.outerRadius ), cohesion_weight )
		const separation = this.multiplyVector( this.separationVector(this.S.conf.innerRadius ), separation_weight )
		
		// Add your code to combine the particle's current this.dir with the (weighted)
		// alignment, cohesion, and separation directions. 
		// Make sure to update the properties this.dir and this.pos accordingly.
		// What happens when the new position lies across the field boundary? 
		
		this.pos = this.pos
		this.dir = this.dir 	
		
	}
	
}


/* =========  you don't need to touch this bit. */

// Reading inputs from the console, helper function
function getInput( name, defaultValue, type = undefined ){
	
	if( !args.hasOwnProperty(name)) return defaultValue
	if( type === "boolean" ){ return true }
	else if( type === "int" ){
		if( args[name][0] == "m" ){
			return -parseInt(args[name].substring(1) )
		}
		return parseInt( args[name] ) 
	} 
	else if (type === "float" ){ return parseFloat( args[name] ) } 
	else { return args[name] }
}

const fieldsize = getInput( "f", 400, "int" ) // field size
const Nboids = getInput( "N", 50, "int" ) // number of boids
const inR = getInput( "i", 10, "int" ) // inner radius
const oR = getInput( "o", 25, "int" ) // outer radius
const wc = getInput( "c", 1, "float" ) // cohesion weight
const ws = getInput( "s", 1, "float" ) // separation weight
const wa = getInput( "a", 1, "float" ) // alignment weight
const runTime = getInput( "T", 1000, "int" ) // number of steps to run
const imgpath = getInput( "I", undefined, "string" ) // path to write images to

let saveImg = false

if( imgpath != undefined ){
	saveImg = true
	fs.mkdirSync(imgpath, { recursive: true })
}


let canvas, S
let conf = {
	w : fieldsize,
	h : fieldsize,
	N : Nboids, 
	zoom : 1,
	innerRadius : inR,
	outerRadius : oR,
	cohesion : wc,
	separation : ws,
	alignment : wa,
	runTime : runTime
}



class Canvas {
	constructor( Scene, conf ){
		this.zoom = conf.zoom
		this.S = Scene
		this.height = this.S.h
		this.width = this.S.w
		
		if( typeof document !== "undefined" ){
			this.el = document.createElement("canvas")
			this.el.width = this.width*this.zoom
			this.el.height = this.height*this.zoom
			let parent_element = document.getElementById( "canvasModel" ) 
			parent_element.appendChild( this.el )
		} else {
			const {createCanvas} = require("canvas")
			this.el = createCanvas( this.width*this.zoom,
				this.height*this.zoom )
			this.fs = require("fs")
		}
		
		
		this.ctx = this.el.getContext("2d")
		this.ctx.lineWidth = .2
		this.ctx.lineCap="butt"
		
	}
	
	background( col ){
		col = col || "000000"
		this.ctx.fillStyle="#"+col
		this.ctx.fillRect( 0,0, this.el.width, this.el.height )
	}
	
	fillCircle( pos, col ){
		this.ctx.fillStyle="#"+col
		this.ctx.beginPath()
		this.ctx.arc( pos[0], pos[1], this.S.conf.innerRadius/2, 0, 2 * Math.PI )
		this.ctx.stroke()
		this.ctx.fill()
	}
	
	drawCircle( pos, col, radius ){
		this.ctx.lineWidth = .5*this.zoom
		this.ctx.strokeStyle="#"+col
		this.ctx.beginPath()
		this.ctx.arc( pos[0], pos[1], radius * this.zoom, 0, 2 * Math.PI )
		this.ctx.stroke()
	}
	
	drawDirections(){
		this.ctx.strokeStyle="#000000"
		const ctx = this.ctx, zoom = this.zoom
		ctx.beginPath()
		ctx.lineWidth = 2*zoom
	
		for( let p of this.S.swarm ){
			
			const startPoint = p.multiplyVector( p.pos, zoom )
			
			let drawVec = p.multiplyVector( p.dir, this.S.conf.innerRadius * 1.2 * zoom )
			drawVec = p.addVectors( startPoint, drawVec )
				
			ctx.moveTo( startPoint[0], startPoint[1] )
			ctx.lineTo( drawVec[0], drawVec[1] )
		}
		ctx.stroke()		
	}
	
	drawSwarm(){
		this.background( "eaecef" )
		for( let p of this.S.swarm ){
			this.fillCircle( p.pos, "ff0000" )
			this.drawCircle( p.pos, "aaaaaa", this.S.conf.outerRadius )
		}
		this.drawDirections()
	}	
	
	writePNG( fname ){
 
		try {
			this.fs.writeFileSync(fname, this.el.toBuffer())
		}
		catch (err) {
			if (err.code === "ENOENT") {
				let message = "Canvas.writePNG: cannot write to file " + fname +
					", are you sure the directory exists?"
				throw(message)
			}
		}
 
	}
}


class Scene {
	
	constructor( conf ){
		this.w = conf.w
		this.h = conf.h 
		this.conf = conf
		this.swarm = []
		this.makeSwarm()
		this.time = 0
	}
	
	reset(){
		this.swarm = []
		this.time = 0
		this.makeSwarm()
	}
	
	getAngles(){
		let angles = []
		for( let p of this.swarm ){
			const ang = 180 + (180/Math.PI) * Math.atan2( p.dir[1], p.dir[0] )
			angles.push( ang )
		}
		return angles
	}
	
	wrap( pos, reference = undefined ){
	
		// wrapping without a reference: just make sure the coordinate falls within 
		// the space
		if( (typeof reference == 'undefined') ){
		
			if( pos[0] < 0 ) pos[0] += this.w
			if( pos[1] < 0 ) pos[1] += this.h
			if( pos[0] > this.w ) pos[0] -= this.w
			if( pos[1] > this.h ) pos[1] -= this.h
	
			return pos
		}

		// otherwise: make coordinates consistent compared to a reference position
		// we don't want to overwrite the 'pos' object itself (!JavaScript) so deep copy it first
		const pos2 = pos.slice()
		let dx =  pos2[0] - reference[0] , dy = pos2[1] - reference[1]
		if( dx > this.w/2 ) pos2[0] -= this.w
		if( dx < -this.w/2 ) pos2[0] += this.w
		if( dy > this.h/2 ) pos2[1] -= this.h
		if( dy < -this.h/2 ) pos2[1] += this.h
		
		
		return pos2
		
	}
	
	addParticle(){
		const i = this.swarm.length + 1
		this.swarm.push( new Particle( this, i ) )
	}
	
	makeSwarm(){
		for( let i = 0; i < this.conf.N; i++ ) this.addParticle()
	}
	
	randomPosition(){
		let x = Math.random() * this.w
		let y = Math.random() * this.h
		return [x,y]
	}
	
	randomDirection( dim = 2 ){
		let dir = []
		while(dim-- > 0){
			dir.push(this.sampleNorm())
		}
		this.normalizeVector(dir)
		return dir
	}
	
	normalizeVector( a ){
	
		if( a[0] == 0 & a[1] == 0 ) return [0,0]
	
		let norm = 0
		for( let i = 0 ; i < a.length ; i ++ ){
			norm += a[i]*a[i]
		}
		norm = Math.sqrt(norm)
		for( let i = 0 ; i < a.length ; i ++ ){
			a[i] /= norm
		}
		return a
	}
	
	sampleNorm(mu=0, sigma=1) {
		let u1 = Math.random()
		let u2 = Math.random()
		let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(Math.PI*2 * u2)
		return z0 * sigma + mu
	}
	
	dist( pos1, pos2 ){
	
		let dx = pos1[0] - pos2[0]
		if( dx > this.w/2 ){ dx -= this.w }
		if( dx < (-this.w/2) ){ dx += this.w }
		
		let dy = pos1[1] - pos2[1]
		if( dy > this.h/2 ){ dy -= this.h }
		if( dy < ( -this.h/2 ) ){ dy += this.h }
		
		const dist = Math.sqrt( dx * dx + dy * dy )
		return dist
		//const dist = Math.hypot(pos2[0] - pos1[0], pos2[1] - pos1[1] )	
	}
	
	neighbours( x, distanceThreshold ){
		let r = []
		for( let p of this.swarm ){
			if( p.id == x.id ) continue 
			
			if( this.dist( p.pos, x.pos ) <= distanceThreshold ){
				r.push( p )
			}
		}
		return r
	}	
	
	step(){
		for( let p of this.swarm ){
			p.updateVector()
		}
		this.time++
	}
}


S = new Scene( conf )
canvas = new Canvas( S, conf )
canvas.drawSwarm()

if( saveImg ) canvas.writePNG( `${imgpath}/boids-t${S.time}.png` )

console.log( "time,id,x,y")

for( let t = 0; t <= conf.runTime; t++ ){
	if( saveImg & t % 100 == 0 ){
		canvas.writePNG( `${imgpath}/boids-t${S.time}.png` )
	}
	
	S.step()
	canvas.drawSwarm()
	
	// log positions to the console
	for( let p of S.swarm ){
		let log = [ S.time, p.id, p.pos[0], p.pos[1] ]
		console.log( log.join(",") )
	}
	
}

	


