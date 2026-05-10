# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from llamafactory.data.fault_tolerant import patch_default_pg_timeout
from llamafactory.train.tuner import run_exp  # use absolute import

# Bump torch.distributed default process-group timeout BEFORE any
# torch.distributed init happens (deepspeed / accelerate both lazy-init shortly
# after import). The default is 600 s, which is shorter than the time a
# per-rank DataLoader rebuild can take after a worker SIGKILL. With
# FaultTolerantDataLoader the rank that lost a worker rebuilds in seconds; we
# just need peer ranks to wait long enough at the next collective for the
# rebuilt rank to catch up.
patch_default_pg_timeout(seconds=3600)


def launch():
    run_exp()


if __name__ == "__main__":
    launch()
