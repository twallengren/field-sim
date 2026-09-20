import type { Snapshot, WorkerCommand, WorkerResponse } from '../contracts';
import { Simulation } from './simulation';

let simulation: Simulation | undefined;
let activeGeneration: number | undefined;

/** Process one command. Stale commands are intentionally discarded. */
export function processCommand(command: WorkerCommand): WorkerResponse | undefined {
  try {
    if (command.type === 'init') {
      const replacement = new Simulation(command.setup);
      simulation = replacement;
      activeGeneration = command.generation;
      return { type: 'snapshot', generation: command.generation, snapshot: replacement.snapshot() };
    }

    if (simulation === undefined || activeGeneration !== command.generation) return undefined;
    switch (command.type) {
      case 'advance':
        // Keep one worker message bounded even if a caller supplies a very
        // large batch, so a queued reset/generation change is reached quickly.
        simulation.advance(command.duration, Math.min(command.maxSteps, 100));
        break;
      case 'step':
        simulation.step();
        break;
      case 'parameters':
        simulation.setParameters(command.parameters);
        break;
      case 'paint':
        simulation.paint(command.brush);
        break;
    }
    return { type: 'snapshot', generation: command.generation, snapshot: simulation.snapshot() };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { type: 'error', generation: command.generation, message };
  }
}

function transferables(snapshot: Snapshot): Transferable[] {
  return Object.values(snapshot.fields).map((values) => values.buffer);
}

interface WorkerScope {
  onmessage: ((event: MessageEvent<WorkerCommand>) => void) | null;
  postMessage(message: WorkerResponse, transfer: Transferable[]): void;
}

const workerScope = globalThis as unknown as WorkerScope;
workerScope.onmessage = (event): void => {
  const response = processCommand(event.data);
  if (response === undefined) return;
  workerScope.postMessage(
    response,
    response.type === 'snapshot' ? transferables(response.snapshot) : [],
  );
};
