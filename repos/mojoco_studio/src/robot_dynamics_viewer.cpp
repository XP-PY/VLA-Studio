#include "dynamics_experiments.h"
#include <SDL2/SDL.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {
using Model=std::unique_ptr<mjModel,decltype(&mj_deleteModel)>;
using Data=std::unique_ptr<mjData,decltype(&mj_deleteData)>;
struct Simulation {
  Model model{nullptr,mj_deleteModel};
  Data data{nullptr,mj_deleteData};
  double initial_energy=0;
};

// Separate models matter: actuator masks and passive damping live in mjModel.
// Separate data alone would accidentally share these experiment settings.
class Experiments {
 public:
  std::array<Simulation,8> simulations;
  int ticks=0;

  Experiments() {
    for (auto& simulation:simulations) {
      char error[1024]{};
      simulation.model.reset(mj_loadXML("assets/dynamics_arm.xml",nullptr,error,sizeof(error)));
      if (!simulation.model) throw std::runtime_error(error);
      if (simulation.model->nq!=2 || simulation.model->nv!=2 || simulation.model->nu!=4)
        throw std::runtime_error("Expected dynamics_arm.xml with two joints and four actuators");
      // Brighter headlight changes rendering only; physics remains identical.
      for (int channel=0;channel<3;++channel) {
        simulation.model->vis.headlight.ambient[channel]=.5f;
        simulation.model->vis.headlight.diffuse[channel]=.7f;
      }
      simulation.data.reset(mj_makeData(simulation.model.get()));
      if (!simulation.data) throw std::runtime_error("Could not allocate MuJoCo data");
    }
    reset();
  }

  void refresh(int i) {
    experiments::mode=experiments::descriptions[i].mode;
    auto& sim=simulations[i];
    mj_forward(sim.model.get(),sim.data.get());
  }

  void reset() {
    ticks=0;
    for (int i=0;i<8;++i) {
      auto& sim=simulations[i];
      experiments::reset(sim.model.get(),sim.data.get(),experiments::descriptions[i].mode);
      refresh(i);
      sim.initial_energy=dynamics::energy({sim.data->qpos[0],sim.data->qpos[1]},
                                         {sim.data->qvel[0],sim.data->qvel[1]});
    }
  }

  /**
   * @brief Advance all eight independent initial-value problems by one RK4 step.
   * Each uses dt=model.opt.timestep=0.001 s and the exact shared batch callback.
   * Select its mode BEFORE mj_forward/mj_step, including every RK4 substage.
   * Integer ticks keep all modes at the same simulation time; wall-clock speed
   * changes only how many fixed steps are scheduled, never the integration dt.
   * Stop at 6000 ticks (six seconds), matching the CSV experiments.
   */
  void step() {
    if (ticks>=experiments::steps) return;
    for (int i=0;i<8;++i) {
      refresh(i);
      auto& sim=simulations[i];
      mj_step(sim.model.get(),sim.data.get());
    }
    ++ticks;
  }

  /**
   * @brief Compare live integration to each existing batch CSV at all 601 samples.
   * Both paths start from identical q,dq and use the same fixed RK4 step. Check
   * time (s), angles (rad), rates (rad/s), torque (N m), energy (J) and error (rad)
   * against the corresponding logged quantities with absolute tolerance 1e-8.
   * The tolerance is a numerical regression bound for this model, not a mixed-
   * unit physical norm. Nonfinite values and missing/incomplete CSVs fail.
   * No graphics context is needed. Run robot_dynamics first to produce the CSVs.
   */
  void check() {
    std::array<std::ifstream,8> inputs;
    for (int i=0;i<8;++i) {
      const std::string path=std::string("build/dynamics/")+experiments::descriptions[i].name+".csv";
      inputs[i].open(path);
      if (!inputs[i]) throw std::runtime_error("Missing "+path+"; run ./build/robot_dynamics first");
      std::string header; std::getline(inputs[i],header);
    }
    double worst=0;
    reset();
    for (int tick=0;tick<=experiments::steps;++tick) {
      if (tick%10==0) {
        for (int i=0;i<8;++i) {
          refresh(i);
          const auto* d=simulations[i].data.get();
          const std::array<double,9> actual={d->time,d->qpos[0],d->qpos[1],d->qvel[0],d->qvel[1],
            d->qfrc_actuator[0],d->qfrc_actuator[1],
            dynamics::energy({d->qpos[0],d->qpos[1]},{d->qvel[0],d->qvel[1]}),
            std::hypot(experiments::target[0]-d->qpos[0],experiments::target[1]-d->qpos[1])};
          std::string line;
          if (!std::getline(inputs[i],line)) throw std::runtime_error("Incomplete batch trajectory");
          std::istringstream row(line);
          for (double value:actual) {
            std::string cell;
            if (!std::getline(row,cell,',')) throw std::runtime_error("Incomplete batch CSV row");
            const double expected=std::stod(cell),error=std::abs(value-expected);
            if (!std::isfinite(value) || !std::isfinite(expected) || error>1e-8)
              throw std::runtime_error(std::string("Viewer/batch mismatch: ")+experiments::descriptions[i].name);
            worst=std::max(worst,error);
          }
        }
      }
      if (tick<experiments::steps) step();
    }
    reset();
    for (const auto& sim:simulations)
      if (sim.data->time!=0 || sim.data->qpos[0]!=.6 || sim.data->qpos[1]!=-.8)
        throw std::runtime_error("Replay did not reset the initial state");
    std::cout << "Viewer matches all 8 batch trajectories (601 samples each); max error="
              << worst << ". Replay reset passed.\n";
  }
};

struct Window {
  SDL_Window* window=nullptr;
  SDL_GLContext context=nullptr;
  Window() {
    if (SDL_Init(SDL_INIT_VIDEO)!=0) throw std::runtime_error(SDL_GetError());
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,24);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,1);
    window=SDL_CreateWindow("Week 7 - MuJoCo dynamics: eight experiments",
      SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,1600,1000,
      SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE|SDL_WINDOW_ALLOW_HIGHDPI);
    if (window) context=SDL_GL_CreateContext(window);
    if (!context) {
      const std::string error=SDL_GetError();
      if (window) SDL_DestroyWindow(window);
      SDL_Quit();
      throw std::runtime_error("Cannot create OpenGL viewer: "+error);
    }
    SDL_GL_SetSwapInterval(1);
  }
  ~Window() { SDL_GL_DeleteContext(context); SDL_DestroyWindow(window); SDL_Quit(); }
};

class Renderer {
  mjvScene scene_{};
  mjrContext context_{};
  mjvOption option_{};
 public:
  mjvCamera camera{};
  explicit Renderer(const mjModel* model) {
    mjv_defaultScene(&scene_); mjr_defaultContext(&context_); mjv_defaultOption(&option_);
    mjv_makeScene(model,&scene_,1000);
    mjr_makeContext(model,&context_,mjFONTSCALE_100);
    reset_camera();
  }
  ~Renderer() { mjr_freeContext(&context_); mjv_freeScene(&scene_); }

  void reset_camera() {
    mjv_defaultCamera(&camera);
    camera.lookat[0]=.1; camera.lookat[1]=0; camera.lookat[2]=1;
    camera.distance=2.6; camera.azimuth=90; camera.elevation=-8;
  }

  /**
   * @brief Draw the desired servo pose as two thin green visual-only capsules.
   * World shoulder p0=(0,0,1); elbow p1=p0+L1(cos q1,0,sin q1);
   * tip p2=p1+L2(cos(q1+q2),0,sin(q1+q2)), lengths in m, target in rad.
   * This is the Chapter 4 planar FK mapped to our vertical world x-z plane.
   * Scene geoms do not enter mjModel or produce collisions, mass or forces.
   * Only servo modes have a desired pose; the other five omit this marker.
   */
  void target_pose() {
    const double q1=experiments::target[0],q12=q1+experiments::target[1];
    const mjtNum p0[3]={0,0,1};
    const mjtNum p1[3]={.45*std::cos(q1),0,1+.45*std::sin(q1)};
    const mjtNum p2[3]={p1[0]+.35*std::cos(q12),0,p1[2]+.35*std::sin(q12)};
    const float color[4]={.25f,.95f,.45f,.55f};
    for (int i=0;i<2;++i) {
      auto* geom=&scene_.geoms[scene_.ngeom++];
      mjv_initGeom(geom,mjGEOM_CAPSULE,nullptr,nullptr,nullptr,color);
      mjv_connector(geom,mjGEOM_CAPSULE,.008,i==0?p0:p1,i==0?p1:p2);
    }
  }

  // Mark the fixed shoulder and moving elbow using MuJoCo's computed body poses.
  // These spheres belong only to the render scene, like the target pose above.
  void joint_markers(const mjData* d) {
    const mjtNum size[3]={.025,.025,.025};
    const float color[4]={.85f,.9f,.95f,1};
    for (int body=1;body<=2;++body)
      mjv_initGeom(&scene_.geoms[scene_.ngeom++],mjGEOM_SPHERE,size,d->xpos+3*body,nullptr,color);
  }

  void panel(Experiments& runs,int index,mjrRect viewport) {
    runs.refresh(index);
    const auto& sim=runs.simulations[index];
    const auto& description=experiments::descriptions[index];
    auto* d=sim.data.get();
    mjv_updateScene(sim.model.get(),d,&option_,nullptr,&camera,mjCAT_ALL,&scene_);
    joint_markers(d);
    const bool servo=experiments::is_servo(description.mode);
    if (servo) target_pose();
    mjr_render(viewport,&scene_,&context_);
    char title[256];
    std::snprintf(title,sizeof(title),"%d  %s\n%s",index+1,description.name,description.law);
    mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,viewport,title,nullptr,&context_);
    const double energy=dynamics::energy({d->qpos[0],d->qpos[1]},{d->qvel[0],d->qvel[1]});
    char status[512],error_line[100]="";
    if (servo) std::snprintf(error_line,sizeof(error_line),"\nTarget error %.4f rad | green = target",
      std::hypot(experiments::target[0]-d->qpos[0],experiments::target[1]-d->qpos[1]));
    std::snprintf(status,sizeof(status),
      "%s\nt %.2f s | E %.2f J | dE %+.2f J\nq [%.2f, %.2f] rad | speed %.2f rad/s%s",
      description.observation,d->time,energy,energy-sim.initial_energy,d->qpos[0],d->qpos[1],
      std::hypot(d->qvel[0],d->qvel[1]),error_line);
    mjr_overlay(mjFONT_NORMAL,mjGRID_BOTTOMLEFT,viewport,status,nullptr,&context_);
  }

  void draw(Experiments& runs,int focus,int width,int height,bool paused,double speed) {
    mjr_rectangle({0,0,width,height},.08f,.09f,.12f,1);
    constexpr int header=64;
    if (focus<0) {
      for (int i=0;i<8;++i) {
        const int w=width/4,h=(height-header)/2;
        panel(runs,i,{(i%4)*w+2,(1-i/4)*h+2,w-4,h-4});
      }
    } else panel(runs,focus,{0,0,width,height-header});
    char help[512];
    std::snprintf(help,sizeof(help),
      "%s | %.3gx speed | 0: all eight   1-8: focus   Space: pause/play   R: replay   Right: step 10 ms\n"
      "[ / ]: slower/faster   C: reset camera   Mouse: left rotate / right pan / wheel zoom   Esc: quit",
      runs.ticks==experiments::steps?"Finished (R to replay)":(paused?"Paused":"Playing"),speed);
    mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,{0,height-header,width,header},help,nullptr,&context_);
  }

  void screenshot(const std::string& path,int width,int height) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width)*height*3);
    mjr_readPixels(pixels.data(),nullptr,{0,0,width,height},&context_);
    std::ofstream out(path,std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write screenshot "+path);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (int row=height-1;row>=0;--row)
      out.write(reinterpret_cast<const char*>(pixels.data()+row*width*3),width*3);
  }
};

int selected_mode(const std::string& value) {
  if (value=="all" || value=="0") return -1;
  for (int i=0;i<8;++i)
    if (value==experiments::descriptions[i].name || value==std::to_string(i+1)) return i;
  throw std::runtime_error("Unknown mode: "+value+" (use all, 1-8, or a batch mode name)");
}

void help() {
  std::cout << "Usage: ./build/robot_dynamics_viewer [--mode all|1..8|NAME] [--check] [--smoke-test]\n"
    "Default: live eight-panel comparison at half speed; stops at six simulation seconds.\n"
    "0 all modes | 1-8 focus | Space pause/play | R replay | Right step 10 ms\n"
    "[ / ] slower/faster | C reset camera | Mouse rotate/pan/zoom | Esc quit\n"
    "--check: compare all live trajectories with batch CSVs, without a display.\n"
    "--smoke-test: render all eight modes and individual views, save PPM screenshots, then exit.\n";
  for (int i=0;i<8;++i) std::cout << i+1 << "  " << experiments::descriptions[i].name << '\n';
}
}  // namespace

int main(int argc,char** argv) {
  try {
    int focus=-1; bool check=false,smoke=false;
    for (int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if (arg=="--help" || arg=="-h") { help(); return 0; }
      if (arg=="--check") check=true;
      else if (arg=="--smoke-test") smoke=true;
      else if (arg=="--mode" && i+1<argc) focus=selected_mode(argv[++i]);
      else throw std::runtime_error("Unknown or incomplete argument: "+arg+" (use --help)");
    }
    experiments::CallbackGuard callback;
    Experiments runs;
    if (check) { runs.check(); return 0; }
    Window window;
    Renderer renderer(runs.simulations[0].model.get());
    bool quit=false,paused=false;
    double speed=.5,accumulator=0;
    auto previous=std::chrono::steady_clock::now();
    int smoke_frame=0;
    if (smoke) { for (int i=0;i<2000;++i) runs.step(); paused=true; }
    while (!quit) {
      const auto now=std::chrono::steady_clock::now();
      const double elapsed=std::min(.1,std::chrono::duration<double>(now-previous).count());
      previous=now;
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        if (event.type==SDL_QUIT) quit=true;
        if (event.type==SDL_KEYDOWN && !event.key.repeat) {
          const auto key=event.key.keysym.sym;
          if (key==SDLK_ESCAPE) quit=true;
          else if (key>=SDLK_0 && key<=SDLK_8) focus=key-SDLK_0-1;
          else if (key==SDLK_SPACE) { paused=!paused; accumulator=0; }
          else if (key==SDLK_r) { runs.reset(); paused=false; accumulator=0; }
          else if (key==SDLK_RIGHT) { paused=true; accumulator=0; for (int i=0;i<10;++i) runs.step(); }
          else if (key==SDLK_LEFTBRACKET) speed=std::max(.125,speed/2);
          else if (key==SDLK_RIGHTBRACKET) speed=std::min(2.0,speed*2);
          else if (key==SDLK_c) renderer.reset_camera();
        }
        const mjModel* model=runs.simulations[0].model.get();
        if (event.type==SDL_MOUSEWHEEL)
          mjv_moveCamera(model,mjMOUSE_ZOOM,0,-.05*event.wheel.y,&renderer.camera);
        if (event.type==SDL_MOUSEMOTION && event.motion.state) {
          int width,height; SDL_GetWindowSize(window.window,&width,&height);
          const int panel_height=focus<0?(height-64)/2:height-64;
          const mjtMouse action=(event.motion.state&SDL_BUTTON_RMASK)?mjMOUSE_MOVE_V:mjMOUSE_ROTATE_V;
          if (panel_height>0)
            mjv_moveCamera(model,action,double(event.motion.xrel)/panel_height,
                           double(event.motion.yrel)/panel_height,&renderer.camera);
        }
      }
      if (!paused && runs.ticks<experiments::steps) {
        accumulator+=elapsed*speed;
        const double dt=runs.simulations[0].model->opt.timestep;
        while (accumulator>=dt && runs.ticks<experiments::steps) { runs.step(); accumulator-=dt; }
      }
      int width,height; SDL_GL_GetDrawableSize(window.window,&width,&height);
      if (width<200 || height<200) { SDL_Delay(10); continue; }
      if (smoke) focus=smoke_frame-1;
      renderer.draw(runs,focus,width,height,paused,speed);
      if (smoke) {
        std::filesystem::create_directories("build/dynamics_viewer");
        renderer.screenshot("build/dynamics_viewer/"+std::string(focus<0?"all":experiments::descriptions[focus].name)+".ppm",width,height);
        if (++smoke_frame==9) quit=true;
      }
      SDL_GL_SwapWindow(window.window);
      SDL_Delay(5);
    }
    if (smoke) std::cout << "Rendered grid and all eight focus views: build/dynamics_viewer/*.ppm\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
