# CTPN Demo

This demo allows you to try out text detection using the ctpn model.

## Setup

`cd` into the demo/ folder:

```sh
cd ctpn/demo
```

Install dependencies:

```sh
yarn
```

Build the ctpn model locally which the demo depends on:

```sh
yarn build-deps
```

Launch a development server, and watch files for changes. This command will also automatically open
the demo app in your browser.

```sh
yarn watch
```

## If you are developing the model locally and want to test the changes in the demo

`cd` into the mobilenet/demo folder:

```sh
cd ctpn/demo
```

Rebuild mobilenet locally:
```sh
yarn build-deps
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the `ctpn` source code, just run `yarn build-deps` in the ctpn/demo
folder again.
