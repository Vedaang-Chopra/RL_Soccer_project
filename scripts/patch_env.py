import soccer_twos, os
from pathlib import Path

pkg_path = os.path.dirname(soccer_twos.__file__)
config_path = os.path.join(pkg_path, 'package.py')

with open(config_path, 'r') as f:
    content = f.read()

content = content.replace(
    'if not Path(TRAINING_ENV_PATH).is_file() and not Path(ROLLOUT_ENV_PATH).is_file():',
    'if not Path(TRAINING_ENV_PATH + ".app").is_dir() and not Path(ROLLOUT_ENV_PATH + ".app").is_dir():'
)
content = content.replace(
    'TRAINING_ENV_PATH = "mac_os/soccer-twos.app/Contents/MacOS/UnityEnvironment"',
    'TRAINING_ENV_PATH = "mac_os/soccer-twos"'
)
content = content.replace(
    'ROLLOUT_ENV_PATH = "mac_os/watch-soccer-twos.app/Contents/MacOS/UnityEnvironment"',
    'ROLLOUT_ENV_PATH = "mac_os/watch-soccer-twos"'
)

with open(config_path, 'w') as f:
    f.write(content)
print("Patch applied successfully.")