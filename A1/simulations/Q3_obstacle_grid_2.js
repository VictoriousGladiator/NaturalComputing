const fs = require("fs");
const path = require("path");

const artistooPath = path.join(__dirname, "../artistoo_minimal/build/artistoo-cjs.js");
const CPM = require(artistooPath);

// ======= match your HTML params =======
let OBSTACLE_SPACING = 20;
let OBSTACLE_MARGIN = 5;

// true  = place motile cells INSIDE the obstacle grid region
// false = place motile cells OUTSIDE that region (in the border band)
let PLACEMENT_GRID = true;

let volume = 500;
let radius = Math.sqrt(volume / Math.PI);
let perimeter = 2 * Math.PI * radius;

let N_CELLS = 20;

// Save images to a unique folder per run
const runFolder = path.join(
  __dirname,
  "img_low_framerate",
  `run_${new Date().toISOString().replace(/[:.]/g, "-")}`
);

let config = {
  ndim: 2,
  field_size: [200, 200],

  conf: {
    T: 20,

    // 0 = background, 1 = motile cells, 2 = obstacles
    J: [
      [0, 20, 30],
      [20, 0, 0],
      [30, 0, 0],
    ],

    LAMBDA_V: [0, 50, 500],
    V: [0, volume, volume / 2],

    // match HTML (your Node version had different P/LAMBDA_P)
    LAMBDA_P: [0, 1, 50],
    P: [0, 340, perimeter / 2],

    LAMBDA_ACT: [0, 200, 0],
    MAX_ACT: [0, 20, 0],
    ACT_MEAN: "arithmetic",
  },

  simsettings: {
    NRCELLS: [N_CELLS, 0],
    RUNTIME: 11620, // about a minute on my machine 
    zoom: 4,
    CANVASCOLOR: "eaecef",
    CELLCOLOR: ["ff4d4d", "4d79ff"],
    SHOWBORDERS: [true, true],
    BORDERCOL: ["000000", "000000"],
    ACTCOLOR: [true, false],

    // ======= image saving =======
    SAVEIMG: true,
    IMGFRAMERATE: 1000, // usually means: save every N MCS (every 10 steps)
    SAVEPATH: runFolder,
    EXPNAME: "simulation",
  },
};

if (!fs.existsSync(config.simsettings.SAVEPATH)) {
  fs.mkdirSync(config.simsettings.SAVEPATH, { recursive: true });
  console.log(`Created directory: ${config.simsettings.SAVEPATH}`);
}

class CustomSim extends CPM.Simulation {
  initializeGrid() {
    if (!this.helpClasses["gm"]) this.addGridManipulator();

    const W = this.C.extents[0],
      H = this.C.extents[1];
    const spacing = OBSTACLE_SPACING;
    const margin = OBSTACLE_MARGIN;

    const Vobs = this.C.conf.V[2];
    const rObs = Math.max(2, Math.round(Math.sqrt(Vobs / Math.PI)));

    // 1) Seed obstacles regularly
    for (let x = margin; x < W - margin; x += spacing) {
      for (let y = margin; y < H - margin; y += spacing) {
        const cid = this.gm.seedCellAt(2, [x, y]);
        let vox = [];
        vox = this.gm.makeCircle([x, y], rObs, vox);
        this.gm.assignCellPixels(vox, 2, cid);
      }
    }

    const isInsideObstacleRegion = (x, y) =>
      x >= margin && x < W - margin && y >= margin && y < H - margin;

    // 2) Seed motile cells inside or outside the obstacle region
    let seeded = 0;
    let attempts = 0;
    const maxAttempts = N_CELLS * 500;

    while (seeded < N_CELLS && attempts < maxAttempts) {
      attempts++;

      let x, y;

      if (PLACEMENT_GRID) {
        // INSIDE
        x = margin + Math.floor(Math.random() * (W - 2 * margin));
        y = margin + Math.floor(Math.random() * (H - 2 * margin));
      } else {
        // OUTSIDE (border band)
        x = Math.floor(Math.random() * W);
        y = Math.floor(Math.random() * H);
        if (isInsideObstacleRegion(x, y)) continue;
      }

      // only seed if background
      if (this.C.pixt([x, y]) === 0) {
        this.gm.seedCellAt(1, [x, y]);
        seeded++;
      }
    }

    // fallback
    while (seeded < N_CELLS) {
      this.gm.seedCell(1);
      seeded++;
    }
  }
}

// ---- steps-per-minute benchmark (machine-dependent) ----
function estimateStepsPerMinute(seconds = 3) {
  const sim = new CustomSim(config);

  // small warmup
  for (let i = 0; i < 50; i++) sim.step();

  const t0 = Date.now();
  let steps = 0;

  while (Date.now() - t0 < seconds * 1000) {
    sim.step();
    steps++;
  }

  const stepsPerSec = steps / seconds;
  const stepsPerMin = stepsPerSec * 60;

  console.log(
    `Estimated speed: ~${stepsPerSec.toFixed(1)} steps/sec => ~${stepsPerMin.toFixed(
      0
    )} steps/min (on this machine, with current settings).`
  );
}

estimateStepsPerMinute(3);

// ---- actual run (saves images) ----
const sim = new CustomSim(config);
sim.run();

console.log("Done. Images saved to:", config.simsettings.SAVEPATH);
