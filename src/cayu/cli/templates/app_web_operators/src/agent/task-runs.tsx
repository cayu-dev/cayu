import { useQuery } from "@tanstack/react-query";
import { AppPageHeader, AppPageTitle } from "@kit/layouts/app";
import { Task01Icon } from "@kit/icons";
import type { AppArea } from "../app/area";
import { useAreaApi } from "../app/services";
import { TaskRuns, taskRunsApi, taskRunsQuery } from "../blocks/task-runs";

/**
 * THE AGENT'S OWN. Task runs: the Agent's Cayu tasks, one row per run of its work, from
 * `GET /app/task-runs`. Say what each run did in `summary`, or name the runs of one task type
 * with `taskRunsQuery(api, { type })`; how tasks group into runs is `run_key` in
 * integrations/http/__init__.py.
 */
export const taskRunsArea: AppArea = {
  id: "task-runs",
  label: "Task runs",
  icon: Task01Icon,
  path: "/task-runs",
  routes: () => [{ path: "task-runs", Component: TaskRunsPage }],
};

function TaskRunsPage() {
  const api = useAreaApi(taskRunsApi);
  const runs = useQuery(taskRunsQuery(api));
  return (
    <div className="flex flex-col gap-page-section" data-origin="own">
      <AppPageHeader description="The Agent's background work, newest first, one row per run.">
        <AppPageTitle>Task runs</AppPageTitle>
      </AppPageHeader>
      <TaskRuns
        label="Task runs"
        query={runs}
        itemLabel={{ singular: "run", plural: "runs" }}
        emptyDescription="Runs show here once the Agent queues Cayu tasks."
      />
    </div>
  );
}
