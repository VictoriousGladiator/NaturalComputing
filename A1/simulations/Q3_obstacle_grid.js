const fs = require("fs");
const path = require("path");
const artistooPath = path.join(__dirname, "../artistoo_minimal/build/artistoo-cjs.js");
const CPM = require(artistooPath);

let OBSTACLE_SPACING = 25
let OBSTACLE_MARGIN  = 5
let N_CELLS = 20

let config = {
    ndim : 2,
    field_size : [200,200],

    conf : {
      T : 20,

      // 0 = background, 1 = motile cells, 2 = obstacles
      J: [
        [0, 20, 30],
        [20, 0,  0],
        [30, 0,  0]
      ],

      LAMBDA_V : [0, 50, 500],
      V        : [0, 500, 250],

      LAMBDA_P : [0,  2,  50],
      P        : [0, 340, 56],

      LAMBDA_ACT : [0, 200, 0],
      MAX_ACT    : [0,  20, 0],
      ACT_MEAN   : "arithmetic"
    },

    simsettings : {
      NRCELLS : [N_CELLS, 0],
      RUNTIME : 500,
      zoom : 4,
      CANVASCOLOR : "eaecef",
      CELLCOLOR : ["ff4d4d", "4d79ff"],
      SHOWBORDERS : [true, true],
      BORDERCOL   : ["000000", "000000"],
      ACTCOLOR : [true, false],
      SAVEIMG : true,
	  IMGFRAMERATE : 10,	
	  SAVEPATH : "img",
      EXPNAME : "simulation"
    }
  }

if (!fs.existsSync(config.simsettings.SAVEPATH)) {
    fs.mkdirSync(config.simsettings.SAVEPATH, { recursive: true });
    console.log(`Created directory: ${config.simsettings.SAVEPATH}`);
}

class custom_sim extends CPM.Simulation {
    initializeGrid(){
    if( !this.helpClasses["gm"] ){ this.addGridManipulator() } 

    const W = this.C.extents[0], H = this.C.extents[1]
    const spacing = OBSTACLE_SPACING
    const margin  = OBSTACLE_MARGIN
    const Vobs = this.C.conf.V[2] 
    const rObs = Math.max(2, Math.round(Math.sqrt(Vobs / Math.PI)))

    for( let x = margin; x < W - margin; x += spacing ){
      for( let y = margin; y < H - margin; y += spacing ){
        const cid = this.gm.seedCellAt( 2, [x,y] ) 
        let vox = []
        vox = this.gm.makeCircle( [x,y], rObs, vox )          
        this.gm.assignCellPixels( vox, 2, cid )               
      }
    }
    let seeded = 0
    while( seeded < N_CELLS ){
    const x = margin + Math.floor( Math.random()*(W - 2*margin) )
    const y = margin + Math.floor( Math.random()*(H - 2*margin) )
    if( this.C.pixt([x,y]) === 0 ){
        this.gm.seedCellAt( 1, [x,y] )
        seeded++
    }
    }
    while( seeded < N_CELLS ){
      this.gm.seedCell( 1 )
      seeded++
    }
  }
}
 
let sim = new custom_sim(config);

sim.run()
