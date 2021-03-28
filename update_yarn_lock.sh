update() {
  rm yarn.lock && yarn
  cd ".."
}

for dir in */; do
    cd "$dir"
    echo `pwd`
    update

    cd "$dir""demo"
    if [ $? -eq 0 ]; then
      echo `pwd`
      update
      cd ".."
    fi
done

