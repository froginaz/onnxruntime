// Copyright (c) 2026.
// Implementation of npu_trace_start/stop and the selected tracing backend.
// One file; the active backend is chosen by MYACCEL_TRACE_{CHROME,PERFETTO}.

#include "myaccel/npu.h"
#include "trace.h"

// ===========================================================================
// chrome backend — built-in, dependency-free Chrome Trace Event JSON
// ===========================================================================
#if defined(MYACCEL_TRACE_CHROME)

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>     // _getpid
#define MYACCEL_GETPID _getpid
#else
#include <unistd.h>      // getpid
#define MYACCEL_GETPID getpid
#endif

namespace {

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint64_t this_tid() {
  return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

struct Event {
  const char* name;   // string literal (__func__ / scope name) — stable pointer
  uint64_t    tid;
  int64_t     ts_ns;
  int64_t     dur_ns;
};

struct Recorder {
  std::mutex          mu;
  std::vector<Event>  events;
  std::string         path;
  std::atomic<bool>   enabled{false};
};

Recorder& rec() {
  static Recorder r;
  return r;
}

}  // namespace

namespace myaccel {
namespace trace {

ScopedSlice::ScopedSlice(const char* name) : name_(name), begin_ns_(0) {
  if (rec().enabled.load(std::memory_order_relaxed)) begin_ns_ = now_ns();
}

ScopedSlice::~ScopedSlice() {
  if (begin_ns_ == 0) return;  // tracing was off when the slice opened
  const int64_t end = now_ns();
  Recorder& r = rec();
  std::lock_guard<std::mutex> lk(r.mu);
  r.events.push_back(Event{name_, this_tid(), begin_ns_, end - begin_ns_});
}

}  // namespace trace
}  // namespace myaccel

extern "C" void npu_trace_start(const char* output_path) {
  Recorder& r = rec();
  std::lock_guard<std::mutex> lk(r.mu);
  r.events.clear();
  r.path = output_path ? output_path : "myaccel_trace.json";
  r.enabled.store(true, std::memory_order_relaxed);
}

extern "C" void npu_trace_stop(void) {
  Recorder& r = rec();
  r.enabled.store(false, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lk(r.mu);
  std::FILE* f = std::fopen(r.path.c_str(), "wb");
  if (f == nullptr) return;
  const int pid = MYACCEL_GETPID();
  // Chrome Trace Event format: array of "complete" (ph:"X") events, time in us.
  std::fprintf(f, "[\n");
  for (size_t i = 0; i < r.events.size(); ++i) {
    const Event& e = r.events[i];
    std::fprintf(f,
        "  {\"name\":\"%s\",\"ph\":\"X\",\"pid\":%d,\"tid\":%llu,"
        "\"ts\":%.3f,\"dur\":%.3f,\"cat\":\"npu\"}%s\n",
        e.name, pid, static_cast<unsigned long long>(e.tid),
        e.ts_ns / 1000.0, e.dur_ns / 1000.0,
        (i + 1 < r.events.size()) ? "," : "");
  }
  std::fprintf(f, "]\n");
  std::fclose(f);
  r.events.clear();
}

// ===========================================================================
// perfetto backend — Perfetto Tracing SDK (in-process, file session)
// ===========================================================================
#elif defined(MYACCEL_TRACE_PERFETTO)

#include <fstream>
#include <memory>
#include <string>

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace {
std::unique_ptr<perfetto::TracingSession> g_session;
std::string g_path;

void ensure_initialized() {
  static bool inited = false;
  if (inited) return;
  perfetto::TracingInitArgs args;
  args.backends = perfetto::kInProcessBackend;  // in-app, cross-platform (incl. Windows)
  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();
  inited = true;
}
}  // namespace

extern "C" void npu_trace_start(const char* output_path) {
  ensure_initialized();
  g_path = output_path ? output_path : "myaccel_trace.perfetto-trace";
  perfetto::protos::gen::TrackEventConfig te_cfg;
  te_cfg.add_enabled_categories("npu");
  perfetto::TraceConfig cfg;
  cfg.add_buffers()->set_size_kb(64 * 1024);
  auto* ds = cfg.add_data_sources()->mutable_config();
  ds->set_name("track_event");
  ds->set_track_event_config_raw(te_cfg.SerializeAsString());
  g_session = perfetto::Tracing::NewTrace();
  g_session->Setup(cfg);
  g_session->StartBlocking();
}

extern "C" void npu_trace_stop(void) {
  if (!g_session) return;
  perfetto::TrackEvent::Flush();
  g_session->StopBlocking();
  std::vector<char> data = g_session->ReadTraceBlocking();
  std::ofstream out(g_path, std::ios::binary);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  g_session.reset();
}

// ===========================================================================
// off — no-op session control (instrumentation macros already compile away)
// ===========================================================================
#else

extern "C" void npu_trace_start(const char* /*output_path*/) {}
extern "C" void npu_trace_stop(void) {}

#endif
