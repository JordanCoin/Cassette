# TASK.md

## Current Objective

Initialize Cassette as a production-minded Python system with a strict architectural foundation.

## Build This First

* Python project scaffold
* dependency management
* linting / typing / tests
* initial repo layout
* contracts for:

  * trace
  * event
  * task
* minimal gateway service with:

  * `/healthz`
  * app startup
  * test coverage

## Constraints

* Keep implementation minimal
* Do not add training logic yet
* Do not add orchestration yet
* Do not add UI yet
* Preserve modular architecture for later scale

## Done When

* project installs cleanly
* checks pass
* gateway runs
* schemas exist
* tests pass
