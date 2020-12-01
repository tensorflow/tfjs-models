#!/usr/bin/env bash
# =============================================================================
# Copyright 2019 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

set -e

notify() {
  # $1: the string which encloses the message
  # $2: the message text
  MESSAGE="$1 $2 $1"
  PADDING=$(eval $(echo printf '"$1%0.s"' {1..${#MESSAGE}}))
  SPACING="$1 $(eval $(echo printf '.%0.s' {1..${#2}})) $1"
  echo $PADDING
  echo $SPACING
  echo $MESSAGE
  echo $SPACING
  echo $PADDING
}

elementIn() {
  # Checks whether $1 is in the array $2
  # Example:
  # The snippet
  #   array=("1" "a string" "3")
  #   containsElement "a string" "${array[@]}"
  # Returns 1
  # Source:
  # https://stackoverflow.com/questions/3685970/check-if-a-bash-array-contains-a-value
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

# The mechanics of reading the long options are taken from
# https://mywiki.wooledge.org/ComplexOptionParsing#CA-78a4030cda5dc5c45377a4d98ebd4fe610e0aa7e_2
usage() {
  echo "Usage:"
  echo "  $0 [ --help | -h ]"
  echo "  $0 [ --target_dir=<value> | --target_dir <value> ] [options]"
  echo
  echo "The default target_dir is ./scripts/dist."
}

# set defaults
SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"
LAST_ARG_IDX=$(($# + 1))
TARGET_DIR="$SCRIPT_DIR/dist"
USE_VENV="false"
declare -a LONGOPTS
# Use associative array to declare how many arguments a long option
# expects. In this case we declare that loglevel expects/has one
# argument and range has two. Long options that aren't listed in this
# way will have zero arguments by default.
LONGOPTS=([target_dir]=1)
OPTSPEC="h-:"
while getopts "$OPTSPEC" opt; do
  while true; do
    case "${opt}" in
    -) #OPTARG is name-of-long-option or name-of-long-option=value
      if [[ ${OPTARG} =~ .*=.* ]]; then # with this --key=value format only one argument is possible
        opt=${OPTARG/=*/}
        ((${#opt} <= 1)) && {
          echo "Syntax error: Invalid long option '$opt'" >&2
          exit 2
        }
        if (($((LONGOPTS[$opt])) != 1)); then
          echo "Syntax error: Option '$opt' does not support this syntax." >&2
          exit 2
        fi
        OPTARG=${OPTARG#*=}
      else #with this --key value1 value2 format multiple arguments are possible
        opt="$OPTARG"
        ((${#opt} <= 1)) && {
          echo "Syntax error: Invalid long option '$opt'" >&2
          exit 2
        }
        OPTARG=(${@:OPTIND:$((LONGOPTS[$opt]))})
        ((OPTIND += LONGOPTS[$opt]))
        echo $OPTIND
        ((OPTIND > $LAST_ARG_IDX)) && {
          echo "Syntax error: Not all required arguments for option '$opt' are given." >&2
          exit 3
        }
      fi
      continue
      ;;
    target_dir)
      TARGET_DIR=$(realpath $OPTARG)
      ;;
    h | help)
      usage
      exit 0
      ;;
    ?)
      echo "Syntax error: Unknown short option '$OPTARG'" >&2
      exit 2
      ;;
    *)
      echo "Syntax error: Unknown long option '$opt'" >&2
      exit 2
      ;;
    esac
    break
  done
done

# internal variables
TRUE=(
  "true"
  "t"
  "1"
  "yes"
  "y"
)
TARGET_DIR=$(realpath $TARGET_DIR)

notify "=" "Building the docker images..."
export OPENCV_JS_DIR="$TARGET_DIR/opencv.js"
docker-compose build
docker-compose up

DESTINATION=$(dirname $SCRIPT_DIR)/src/assets/opencv/opencv.js
notify "=" "Copying the distribution file to $DESTINATION..."
cp $OPENCV_JS_DIR/bin/opencv.js $DESTINATION

notify "=" "Success!"
