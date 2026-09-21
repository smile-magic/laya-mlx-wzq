#!/bin/zsh
cd -- "${0:A:h}" || exit 1
./run.sh "$@"
GOMOKU_EXIT=$?
if (( GOMOKU_EXIT != 0 && GOMOKU_EXIT != 130 )) && [[ -t 0 ]]; then
  print '\n启动失败，请查看上方错误。Startup failed; see the error above.'
  read '?按回车关闭 / Press Return to close…'
fi
exit "$GOMOKU_EXIT"
