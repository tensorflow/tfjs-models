## QnA Demo

This demo allows you to find answers of a question from the given context.
To use the demo, you can update the text area with your own text, type your
question into the input box, and click the search button. You will see the
answers displayed in the Answers section.

## Setup

`cd` into the demo/ folder:

```sh
cd qna/demo
```

Install dependencies:

```sh
yarn
```

Launch a development server, and watch files for changes. This command will also automatically open
the demo app in your browser.

```sh
yarn watch
```

## If you are developing the model locally and want to test the changes in the demo

`cd` into the qna/ folder:

```sh
cd qna
```

Install dependencies:
```sh
yarn
```

Publish qna locally:
```sh
yarn publish-local
```

`cd` into this directory (qna/demo) and install dependencies:

```sh
cd demo
yarn
```

Link the package published from the publish step above:
```sh
yarn link-local
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the `qna` source code, just run `yarn publish-local` in the qna/
folder again.
