/* Cayu's DMTCP 4.2 checkpoint barrier. User threads remain suspended while
 * these hooks run. The controller copies files before releasing the source,
 * and commits restoration authority before releasing a restored process.
 * Compile into the immutable workload image; never compile agent input. */
#include <dmtcp.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char root[PATH_MAX];

static void hook(DmtcpEvent_t event, DmtcpEventData_t *data) {
  (void)data;
  if (event == DMTCP_EVENT_INIT) {
    const char *value = getenv("CAYU_SNAPSHOT_ROOT");
    if (!value || value[0] != '/' || strlen(value) > PATH_MAX - 64) _exit(125);
    strcpy(root, value);
  }
  if (event != DMTCP_EVENT_RESUME && event != DMTCP_EVENT_RESTART) return;
  char ready[PATH_MAX], release[PATH_MAX];
  const char *kind = event == DMTCP_EVENT_RESUME ? "capture" : "restore";
  snprintf(ready, sizeof(ready), "%s/%s.ready", root, kind);
  snprintf(release, sizeof(release), "%s/%s.release", root, kind);
  int fd = open(ready, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  if (fd < 0 || write(fd, "1", 1) != 1 || fsync(fd) != 0) _exit(125);
  close(fd);
  /* Deliberately no automatic timeout release. A dead controller must leave
   * the process inactive, rather than resume uncommitted work. */
  while (access(release, F_OK) != 0) usleep(10000);
}

static DmtcpPluginDescriptor_t descriptor = {
  DMTCP_PLUGIN_API_VERSION, DMTCP_PACKAGE_VERSION, "cayu-snapshot-barrier",
  "Cayu", "", "Consistent files/process capture and fenced activation", hook
};
DMTCP_DECL_PLUGIN(descriptor);
