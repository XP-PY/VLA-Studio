#include "reaching_pipeline.h"
#include <SDL2/SDL.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {
using namespace reaching;
struct Window {
  SDL_Window* window=nullptr;
  SDL_GLContext context=nullptr;
  Window() {
    if (SDL_Init(SDL_INIT_VIDEO)!=0) throw std::runtime_error(SDL_GetError());
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,1); SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,24);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,2); SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,1);
    window=SDL_CreateWindow("Week 9 - Classical reaching controller comparisons",SDL_WINDOWPOS_CENTERED,
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
    const Vec p=trajectory::tip(q); const mjtNum c[3]={p[0],0,p[1]};
    const float green[4]={.2f,1,.4f,.6f};
    for (int i=0;i<2;++i) {
      auto* geom=&scene_.geoms[scene_.ngeom++];
      mjv_initGeom(geom,mjGEOM_CAPSULE,nullptr,nullptr,nullptr,green);
      mjv_connector(geom,mjGEOM_CAPSULE,.009,i==0?a:b,i==0?b:c);
    }
  }

  void panel(Experiment& sim,mjrRect viewport) {
    sim.refresh();  // does not sample sensors or recompute feedback
    auto* d=sim.data.get();
    const auto ref=reference(sim.ticks*dt,sim.plan,sim.config);
    mjv_updateScene(sim.model.get(),d,&option_,nullptr,&camera,mjCAT_ALL,&scene_);
    desired_pose(ref.q);
    const mjtNum size[3]={.022,.022,.022};
    const float red[4]={1,.25f,.3f,.8f};
    mjv_initGeom(&scene_.geoms[scene_.ngeom++],mjGEOM_SPHERE,size,
                 sim.plan.world_target.data(),nullptr,red);
    mjr_render(viewport,&scene_,&context_);
    char title[256],status[512];
    std::snprintf(title,sizeof(title),"%s | %.0f Hz | torque limit %.1f N m\n"
      "Green: moving reference | red: Cartesian goal\nSeed %u | %s target",
      sim.config.name.c_str(),1/(sim.config.period_steps*dt),sim.config.torque_limit,sim.seed,
      sim.config.jump?"jump":"quintic");
    const Vec p=trajectory::tip({d->qpos[0],d->qpos[1]});
    std::snprintf(status,sizeof(status),
      "t %.3f s | joint tracking error %.4f rad\nFinal-goal distance %.4f m\n"
      "Torque [%.2f, %.2f] N m | %s\nMeasured velocity [%.2f, %.2f] rad/s\n%s",
      sim.ticks*dt,error_norm(ref.q,{d->qpos[0],d->qpos[1]}),
      std::hypot(p[0]-sim.plan.world_target[0],p[1]-sim.plan.world_target[2]),
      d->qfrc_actuator[0],d->qfrc_actuator[1],sim.effort.raw!=sim.effort.applied?"SATURATED":"not saturated",
      sim.observation.v[0],sim.observation.v[1],
      cartesian::inside({d->qpos[0],d->qpos[1]},cartesian::kPhysicalLimits)?"Inside planning range":"OUTSIDE planning range");
    mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,viewport,title,nullptr,&context_);
    mjr_overlay(mjFONT_NORMAL,mjGRID_BOTTOMLEFT,viewport,status,nullptr,&context_);
  }

  void draw(Experiment& baseline,Experiment& candidate,int width,int height,bool paused,double speed) {
    mjr_rectangle({0,0,width,height},.08f,.09f,.12f,1);
    panel(baseline,{2,2,width/2-4,height-74});
    panel(candidate,{width/2+2,2,width/2-4,height-74});
    char text[384];
    std::snprintf(text,sizeof(text),
      "%s | %.3gx | N/B next/previous case | Space pause | R replay | Right step 1 ms | [/] speed\n"
      "Left drag rotate | Right drag pan | Wheel zoom | C reset camera | Esc quit",
      baseline.ticks==baseline.plan.total_steps?"Finished":(paused?"Paused":"Playing"),speed);
    mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,{0,height-72,width,72},text,nullptr,&context_);
  }

  void screenshot(const std::string& name,int width,int height) {
    std::vector<unsigned char> rgb(static_cast<std::size_t>(width)*height*3);
    mjr_readPixels(rgb.data(),nullptr,{0,0,width,height},&context_);
    std::filesystem::create_directories("build/week9_viewer");
    std::ofstream out("build/week9_viewer/"+name+".ppm",std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write viewer screenshot");
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (int y=height-1;y>=0;--y) out.write(reinterpret_cast<const char*>(rgb.data()+y*width*3),width*3);
  }
};

/** @brief Check live q,dq,observations,raw/applied torque against every batch row.
 * Compare columns in their respective physical units with absolute tolerance
 * 1e-8. Extra refresh() calls emulate rendering; they must not consume sensor
 * noise or change the held torque. Reset then reproduces the first observation.
 * This tests identical seeded control histories independently of render rate.
 */
void check(Experiment& sim) {
  const std::string name=sim.config.name+"_seed"+std::to_string(sim.seed);
  std::ifstream in("build/classical_runs/"+name+".csv");
  if (!in) throw std::runtime_error("Missing "+name+" CSV; run the matching classical_reaching batch first");
  std::string line; std::getline(in,line); sim.reset();
  const auto first=sim.observation.q;
  double worst=0;
  for (int tick=0;tick<=sim.plan.total_steps;++tick) {
    sim.refresh(); sim.refresh();
    if (!std::getline(in,line)) throw std::runtime_error("Incomplete CSV");
    std::istringstream row(line); std::string cell; std::vector<double> values;
    while (std::getline(row,cell,',')) values.push_back(std::stod(cell));
    if (values.size()!=27) throw std::runtime_error("Unexpected CSV schema");
    const auto* d=sim.data.get();
    const std::array<double,13> actual={tick*dt,d->qpos[0],d->qpos[1],d->qvel[0],d->qvel[1],
      sim.observation.q[0],sim.observation.q[1],sim.observation.v[0],sim.observation.v[1],
      sim.effort.raw[0],sim.effort.raw[1],sim.effort.applied[0],sim.effort.applied[1]};
    for (int j=0;j<13;++j) {
      const double expected=values[j==0?0:j+7],error=std::abs(actual[j]-expected);
      if (!std::isfinite(actual[j]) || !std::isfinite(expected) || error>1e-8)
        throw std::runtime_error("Viewer/batch mismatch: "+name);
      worst=std::max(worst,error);
    }
    if (std::abs(values[1]-sim.control_time)>1e-10) throw std::runtime_error("Control-time mismatch");
    if (tick<sim.plan.total_steps) sim.step();
  }
  sim.reset();
  if (sim.ticks!=0 || sim.observation.q!=first) throw std::runtime_error("Replay reset changed sensor sequence");
  std::cout << name << ": viewer/batch match, max error=" << worst << '\n';
}
}  // namespace

int main(int argc,char** argv) {
  try {
    bool verify=false,smoke=false,explicit_case=false;
    std::string selected="target_jump"; unsigned seed=0; Point target=default_target;
    for (int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if (arg=="--check") verify=true;
      else if (arg=="--smoke-test") smoke=true;
      else if (arg=="--case" && i+1<argc) { selected=argv[++i]; explicit_case=true; }
      else if (arg=="--seed" && i+1<argc) {
        const std::string value=argv[++i];
        if (value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
          throw std::invalid_argument("Seed must be a nonnegative integer");
        const auto n=std::stoul(value);
        if (n>std::numeric_limits<unsigned>::max()) throw std::invalid_argument("Seed too large");
        seed=static_cast<unsigned>(n);
      } else if (arg=="--target" && i+3<argc) {
        for (double& x:target) { std::size_t used; const std::string value=argv[++i]; x=std::stod(value,&used);
          if (used!=value.size()) throw std::invalid_argument("Invalid coordinate"); }
      } else if (arg=="--help") {
        std::cout << "Usage: ./build/classical_reaching_viewer [--case NAME] [--seed N] [--target X Y Z]\n"
          "       [--check | --smoke-test]\nBaseline at left; selected case at right. N/B cycle cases, R replay.\n"
          "--check checks all cases at the selected seed unless --case is supplied.\n";
        for (const auto& c:configurations()) std::cout << c.name << '\n';
        return 0;
      } else throw std::invalid_argument("Unknown/incomplete option: "+arg);
    }
    const auto plan=make_plan(target); const auto cases=configurations();
    configuration(selected);  // validate before opening any window
    HeldControlGuard guard;
    if (verify) {
      for (const auto& c:cases) if (!explicit_case || c.name==selected) {
        Experiment sim(c,plan,seed); check(sim);
      }
      return 0;
    }
    Experiment baseline(configuration("baseline"),plan,seed);
    auto candidate=std::make_unique<Experiment>(configuration(selected),plan,seed);
    auto brighten=[](Experiment& sim) {
      for (int c=0;c<3;++c) { sim.model->vis.headlight.ambient[c]=.5f; sim.model->vis.headlight.diffuse[c]=.7f; }
    };
    brighten(baseline); brighten(*candidate);
    Window window; Renderer renderer(baseline.model.get());
    bool quit=false,paused=false; double speed=.5,accumulator=0;
    int case_index=0;
    while (cases[case_index].name!=selected) ++case_index;
    auto previous=std::chrono::steady_clock::now();
    int smoke_frame=0;
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
            case SDLK_r: baseline.reset(); candidate->reset(); accumulator=0; paused=false; break;
            case SDLK_RIGHT: paused=true; accumulator=0; baseline.step(); candidate->step(); break;
            case SDLK_LEFTBRACKET: speed=std::max(.125,speed/2); break;
            case SDLK_RIGHTBRACKET: speed=std::min(2.0,speed*2); break;
            case SDLK_c: renderer.reset_camera(); break;
            case SDLK_n: case SDLK_b:
              case_index=(case_index+(event.key.keysym.sym==SDLK_n?1:int(cases.size())-1))%int(cases.size());
              candidate=std::make_unique<Experiment>(cases[case_index],plan,seed);
              brighten(*candidate); baseline.reset(); accumulator=0; paused=false; break;
          }
        }
        if (event.type==SDL_MOUSEWHEEL)
          mjv_moveCamera(baseline.model.get(),mjMOUSE_ZOOM,0,-.05*event.wheel.y,&renderer.camera);
        if (event.type==SDL_MOUSEMOTION && event.motion.state) {
          int w,h; SDL_GetWindowSize(window.window,&w,&h);
          if (h>0) mjv_moveCamera(baseline.model.get(),(event.motion.state&SDL_BUTTON_RMASK)?mjMOUSE_MOVE_V:mjMOUSE_ROTATE_V,
            double(event.motion.xrel)/h,double(event.motion.yrel)/h,&renderer.camera);
        }
      }
      if (smoke) {
        candidate=std::make_unique<Experiment>(cases[smoke_frame],plan,seed);
        brighten(*candidate); baseline.reset(); paused=true;
        for (int i=0;i<1200;++i) { baseline.step(); candidate->step(); }
      } else if (!paused && baseline.ticks<plan.total_steps) {
        accumulator+=elapsed*speed;
        while (accumulator>=dt && baseline.ticks<plan.total_steps) { baseline.step(); candidate->step(); accumulator-=dt; }
      }
      int width,height; SDL_GL_GetDrawableSize(window.window,&width,&height);
      if (width<200 || height<200) { SDL_Delay(10); continue; }
      renderer.draw(baseline,*candidate,width,height,paused,speed);
      if (smoke) {
        renderer.screenshot(candidate->config.name,width,height);
        if (++smoke_frame==int(cases.size())) quit=true;
      }
      SDL_GL_SwapWindow(window.window); SDL_Delay(5);
    }
    if (smoke) std::cout << "Rendered all cases: build/week9_viewer/*.ppm\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
