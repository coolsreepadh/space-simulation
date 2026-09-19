I built Simulation Cluster, a real-time space simulator written in Python that runs on the GPU. It simulates gravity between thousands of particles and renders a black hole that bends light the way general relativity says it should.

🌌 SCENES
• Solar System: the Sun, all 8 planets and 5 dwarf planets on their real orbits, with the correct tilt and orbit shape. It also has major moons, Saturn's rings, the asteroid and Kuiper belts, and real space mission trajectories. Pick any date and every planet goes to where it actually was, or will be, on that day. The Oort cloud and heliosphere can be switched on too.
• Galaxy Merger: two spiral galaxies collide. You can watch the tidal tails and the bridge between them form, then the merger itself.
• Black Hole: a Schwarzschild black hole. Drop a star into it and watch tidal forces shred it into a stream that settles into an accretion disk.
• Star Cluster Collapse: a cold cluster of stars collapses under its own gravity, with every star pulling on every other star.

⚙️ HOW IT WORKS
• Physics: an N-body gravity integrator running on the GPU with Taichi (CUDA)
• Black hole rendering: light rays are ray-marched along curved paths (null geodesics) around the black hole, so the simulated stars behind it get gravitationally lensed too
• Graphics: ModernGL with HDR rendering, bloom and ACES tone mapping
• UI: Dear ImGui control panel for changing scenes and parameters live

🛠️ BUILT WITH
Python · Taichi · ModernGL · Pygame · NumPy · Dear ImGui

