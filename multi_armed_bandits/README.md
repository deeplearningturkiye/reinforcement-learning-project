# Multi Armed Bandits

## Ten Armed Testbed

```
from ten_armed_testbed import ten_armed_testbed

# These are default values, if used this way, there is no need to specify
tat = ten_armed_testbed(variance=1, min=-2, max=2) 

# Pulls the first arm
reward = tat.pullArm(0)

# Pulls the last arm
reward = tat.pullArm(9)

```