#include "trajectory_experiment.h"
#include <SDL2/SDL.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {
using namespace trajectory;
struct Window {
  SDL_Window* window=nullptr;
  SDL_GLContext context=nullptr;
  Window() {
    if (SDL_Init(SDL_INIT_VIDEO)!=0) throw std::runtime_error(SDL_GetError());
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,1); SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,24);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,2); SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,1);
    window=SDL_CreateWindow("Week 8 - Cubic and quintic trajectory tracking",SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED,1400,850,SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE|SDL_WINDOW_ALLOW_HIGHDPI);
    if (window) context=SDL_GL_CreateContext(window);
    if (!context) {
      const std::string error=SDL_GetError();
      if (window) SDL_DestroyWindow(window);
      SDL_Quit(); throw std::runtime_error(error);
    }
    SDL_SetWindowMinimumSize(window,800,500);
    SDL_GL_SetSwapInterval(1);
  }
  ~Window() { SDL_GL_DeleteContext(context); SDL_DestroyWindow(window); SDL_Quit(); }
};

class Renderer {
  mjvScene scene_{}; mjrContext context_{}; mjvOption option_{};
 public:
  mjvCamera camera{};
  explicit Renderer(const mjModel* m) {
    mjv_defaultScene(&scene_); mjr_defaultContext(&context_); mjv_defaultOption(&option_);
    mjv_makeScene(m,&scene_,1000); mjr_makeContext(m,&context_,mjFONTSCALE_150);
    reset_camera();
  }
  ~Renderer() { mjr_freeContext(&context_); mjv_freeScene(&scene_); }
  void reset_camera() {
    mjv_defaultCamera(&camera); camera.lookat[0]=.35; camera.lookat[2]=1;
    camera.distance=2.1; camera.azimuth=90; camera.elevation=-8;
  }

  /** @brief Visual-only desired pose via planar FK in world x-z coordinates.
   * p0=(0,0,1), p1=p0+L1(cos q1,0,sin q1), p2 from tip(q).
   * q is the current reference in radians, lengths in metres. Green capsules
   * are render annotations with no physical mass/contact, not a second robot.
   * Modern Robotics notes 9.2.1: FK maps the joint path into Cartesian space.
   */
  void desired_pose(const Vec& q) {
    const mjtNum a[3]={0,0,1},b[3]={.45*std::cos(q[0]),0,1+.45*std::sin(q[0])};
    const Vec p=tip(q); const mjtNum c[3]={p[0],0,p[1]};
    const float green[4]={.2f,1,.4f,.6f};
    for (int i=0;i<2;++i) {
      auto* geom=&scene_.geoms[scene_.ngeom++];
      mjv_initGeom(geom,mjGEOM_CAPSULE,nullptr,nullptr,nullptr,green);
      mjv_connector(geom,mjGEOM_CAPSULE,.009,i==0?a:b,i==0?b:c);
    }
  }

  void panel(Experiment& sim,mjrRect viewport) {
    sim.refresh();
    auto* d=sim.data.get();
    const auto ref=reference(d->time,sim.profile);
    mjv_updateScene(sim.model.get(),d,&option_,nullptr,&camera,mjCAT_ALL,&scene_);
    desired_pose(ref.q);
    mjr_render(viewport,&scene_,&context_);
    const char* law=sim.profile==Profile::Cubic?"s = 3u^2 - 2u^3":"s = 10u^3 - 15u^4 + 6u^5";
    const char* phase=d->time<hold_before?"initial hold":(d->time<=hold_before+duration?"moving":"final hold");
    char title[256],status[512];
    std::snprintf(title,sizeof(title),"%s | %s\n%s\nGreen = moving reference; solid = actual robot",name(sim.profile),phase,law);
    std::snprintf(status,sizeof(status),
      "t %.3f s | joint error %.5f rad\nDesired velocity [%.3f, %.3f] rad/s\nDesired accel. [%.3f, %.3f] rad/s^2\nActual accel.  [%.3f, %.3f] rad/s^2",
      d->time,std::hypot(ref.q[0]-d->qpos[0],ref.q[1]-d->qpos[1]),
      ref.dq[0],ref.dq[1],ref.ddq[0],ref.ddq[1],d->qacc[0],d->qacc[1]);
    mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,viewport,title,nullptr,&context_);
    mjr_overlay(mjFONT_NORMAL,mjGRID_BOTTOMLEFT,viewport,status,nullptr,&context_);
  }

  void draw(Experiment& cubic,Experiment& quintic,int width,int height,bool paused,double speed) {
    mjr_rectangle({0,0,width,height},.08f,.09f,.12f,1);
    panel(cubic,{2,2,width/2-4,height-74});
    panel(quintic,{width/2+2,2,width/2-4,height-74});
    char text[384];
    std::snprintf(text,sizeof(text),
      "%s | %.3gx | Space pause/play | R replay | Right step 1 ms | [ / ] speed | Esc quit\n"
      "Left drag rotate | Right drag pan | Wheel zoom | C reset camera | Stops at 3.5 s",
      cubic.ticks==Experiment::total_steps?"Finished":(paused?"Paused":"Playing"),speed);
    mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,{0,height-72,width,72},text,nullptr,&context_);
  }

  void screenshot(int width,int height) {
    std::vector<unsigned char> rgb(static_cast<std::size_t>(width)*height*3);
    mjr_readPixels(rgb.data(),nullptr,{0,0,width,height},&context_);
    std::filesystem::create_directories("build");
    std::ofstream out("build/week8_viewer.ppm",std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write viewer screenshot");
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (int y=height-1;y>=0;--y) out.write(reinterpret_cast<const char*>(rgb.data()+y*width*3),width*3);
  }
};

/** @brief Verify viewer simulation against all 3501 batch samples per profile.
 * Check time, q,dq,qdd,tau in their logged units with absolute tolerance 1e-8;
 * the shared Experiment supplies the same initial state, RK4 stages and control.
 * This needs no display. Malformed/missing data and nonfinite values fail.
 */
void check(Experiment& sim) {
  std::ifstream in(std::string("build/trajectories/")+name(sim.profile)+".csv");
  if (!in) throw std::runtime_error("Run ./build/trajectory_tracking before --check");
  std::string line; std::getline(in,line);
  double worst=0;
  sim.reset();
  for (int t=0;t<=Experiment::total_steps;++t) {
    if (!std::getline(in,line)) throw std::runtime_error("Incomplete trajectory CSV");
    std::istringstream row(line); std::vector<double> values; std::string cell;
    while (std::getline(row,cell,',')) values.push_back(std::stod(cell));
    if (values.size()!=23) throw std::runtime_error("Unexpected trajectory columns");
    const auto* d=sim.data.get();
    std::array<double,9> actual={d->time,d->qpos[0],d->qpos[1],d->qvel[0],d->qvel[1],
      d->qacc[0],d->qacc[1],d->qfrc_actuator[0],d->qfrc_actuator[1]};
    for (int j=0;j<9;++j) {
      const double expected=values[j==0?0:j+7],error=std::abs(actual[j]-expected);
      if (!std::isfinite(actual[j]) || !std::isfinite(expected) || error>1e-8)
        throw std::runtime_error("Viewer/batch trajectory mismatch");
      worst=std::max(worst,error);
    }
    if (t<Experiment::total_steps) sim.step();
  }
  sim.reset();
  if (sim.ticks!=0 || sim.data->time!=0 || sim.data->qpos[0]!=start[0] || sim.data->qpos[1]!=start[1])
    throw std::runtime_error("Viewer replay reset failed");
  std::cout << name(sim.profile) << ": viewer/batch check passed, max error=" << worst << '\n';
}
}  // namespace

int main(int argc,char** argv) {
  try {
    bool verify=false,smoke=false;
    for (int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if (arg=="--check") verify=true;
      else if (arg=="--smoke-test") smoke=true;
      else if (arg=="--help") {
        std::cout << "Usage: ./build/trajectory_viewer [--check | --smoke-test]\n"
          "Space pause, R replay, Right step 1 ms, [/] speed, mouse camera, C reset, Esc exit.\n";
        return 0;
      } else throw std::invalid_argument("Unknown argument: "+arg);
    }
    CallbackGuard callback;
    Experiment cubic(Profile::Cubic),quintic(Profile::Quintic);
    if (verify) { check(cubic); check(quintic); return 0; }
    for (auto* sim:{&cubic,&quintic}) for (int c=0;c<3;++c) {
      sim->model->vis.headlight.ambient[c]=.5f;
      sim->model->vis.headlight.diffuse[c]=.7f;
    }
    Window window; Renderer renderer(cubic.model.get());
    bool quit=false,paused=false; double speed=.5,accumulator=0;
    auto previous=std::chrono::steady_clock::now();
    if (smoke) { for (int i=0;i<1200;++i) { cubic.step(); quintic.step(); } paused=true; }
    while (!quit) {
      const auto now=std::chrono::steady_clock::now();
      const double elapsed=std::min(.1,std::chrono::duration<double>(now-previous).count()); previous=now;
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        if (event.type==SDL_QUIT) quit=true;
        if (event.type==SDL_KEYDOWN && !event.key.repeat) {
          switch (event.key.keysym.sym) {
            case SDLK_ESCAPE: quit=true; break;
            case SDLK_SPACE: paused=!paused; accumulator=0; break;
            case SDLK_r: cubic.reset(); quintic.reset(); accumulator=0; paused=false; break;
            case SDLK_RIGHT: paused=true; accumulator=0; cubic.step(); quintic.step(); break;
            case SDLK_LEFTBRACKET: speed=std::max(.125,speed/2); break;
            case SDLK_RIGHTBRACKET: speed=std::min(2.0,speed*2); break;
            case SDLK_c: renderer.reset_camera(); break;
          }
        }
        if (event.type==SDL_MOUSEWHEEL)
          mjv_moveCamera(cubic.model.get(),mjMOUSE_ZOOM,0,-.05*event.wheel.y,&renderer.camera);
        if (event.type==SDL_MOUSEMOTION && event.motion.state) {
          int w,h; SDL_GetWindowSize(window.window,&w,&h);
          if (h>0) mjv_moveCamera(cubic.model.get(),(event.motion.state&SDL_BUTTON_RMASK)?mjMOUSE_MOVE_V:mjMOUSE_ROTATE_V,
            double(event.motion.xrel)/h,double(event.motion.yrel)/h,&renderer.camera);
        }
      }
      if (!paused && cubic.ticks<Experiment::total_steps) {
        accumulator+=elapsed*speed;
        while (accumulator>=.001 && cubic.ticks<Experiment::total_steps) {
          cubic.step(); quintic.step(); accumulator-=.001;
        }
      }
      int width,height; SDL_GL_GetDrawableSize(window.window,&width,&height);
      if (width<200 || height<200) { SDL_Delay(10); continue; }
      renderer.draw(cubic,quintic,width,height,paused,speed);
      if (smoke) { renderer.screenshot(width,height); quit=true; }
      SDL_GL_SwapWindow(window.window); SDL_Delay(5);
    }
    if (smoke) std::cout << "Rendered both trajectories: build/week8_viewer.ppm\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
