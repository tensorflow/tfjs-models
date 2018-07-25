// Type definitions for vorpal 1.12.0
// Project: https://github.com/dthree/vorpal
// Definitions by: Jim Buck <http://github.com/jimmyboh>, Amir Arad
// <greenshade@gmail.com> Definitions:
// https://github.com/DefinitelyTyped/DefinitelyTyped


declare module 'vorpal' {
import {ParsedArgs} from 'minimist';
import {EventEmitter} from 'events';
import {Inquirer} from 'inquirer';

  class Vorpal extends EventEmitter {
    // class Session{};
    // chalk : Chalk;
    //        cmdHistory: {};
    commands: Array<Vorpal.Command>;
    session: Vorpal.Session;
    ui: Vorpal.UiInstance;

    /**
     * Parses the process's `process.argv` arguments and executes the matching
     * command.
     *
     * @param {(string|string[])} argv
     * @param {{use?: 'minimist'}} [options] When `use: 'minimist'` is passed in as an option, `.parse` will instead expose the `minimist` module's main method and return the results, executing no commands.
     */
    parse(argv: string|string[], options?: {use?: 'minimist'}): ParsedArgs;

    /**
     * Sets version of your application's API.
     *
     * @param {String} version
     * @return {Vorpal}
     * @api public
     */
    version(version: string): this;

    /**
     * Sets the permanent delimiter for this
     * Vorpal server instance.
     *
     * @param {String} str
     * @return {Vorpal}
     * @api public
     */
    delimiter(str: string): this;


    /**
     * Imports a library of Vorpal API commands
     * from another Node module as an extension
     * of Vorpal.
     *
     * @param {Function} commands
     * @param  options
     * @return {Vorpal}
     * @api public
     */
    use<T>(
        commands: (this: this, vorpal: this, options?: T) => any,
        options?: T): this;

    use(commandsPath: string, options: any): this;

    use(commands: Vorpal.CommandData|Array<Vorpal.CommandData>): this;

    /**
     * Registers a new command in the vorpal API.
     *
     * @param {String} name
     * @param {String} description
     * @param {Object} options
     * @return {Command}
     * @api public
     */
    command(
        name: string, description?: string,
        options?: Vorpal.CommandOptions): Vorpal.Command;

    /**
     * Registers a new 'mode' command in the vorpal API.
     *
     * @param {String} name
     * @param {String} description
     * @param {Object} options
     * @return {Command}
     * @api public
     */
    mode(name: string, description?: string, options?: Vorpal.CommandOptions):
        Vorpal.Command;

    /**
     * Registers a 'catch' command in the vorpal API.
     * This is executed when no command matches are found.
     *
     * @param {String} name
     * @param {String} description
     * @param {Object} options
     * @return {Command}
     * @api public
     */
    catch(name: string, description?: string, options?: Vorpal.CommandOptions):
        Vorpal.Command;

    /**
     * An alias to the `catch` command.
     *
     * @param {String} name
     * @param {String} description
     * @param {Object} options
     * @return {Command}
     * @api public
     */
    default(
        name: string, description?: string,
        options?: Vorpal.CommandOptions): Vorpal.Command;

    /**
     * Delegates to ui.log.
     *
     * @param {String} log
     * @return {Vorpal}
     * @api public
     */
    log(message?: any, ...optionalParams: any[]): this;

    /**
     * Intercepts all logging through `vorpal.log`
     * and runs it through the function declared by
     * `vorpal.pipe()`.
     *
     * @param {Function} pipeFn
     * @return {Vorpal}
     * @api public
     */
    pipe(pipeFn: (stdout: string) => string): this;

    /**
     * If Vorpal is the local terminal,
     * hook all stdout, through a fn.
     *
     * @return {this}
     * @api private
     */
    hook(hookFn: (stdout: string) => any): this;

    /**
     * Set id for command line history
     * @param id
     * @return {this}
     * @api public
     */
    history(id: string): this;

    /**
     * Set id for local storage
     * @param id
     * @return {this}
     * @api public
     */
    localStorage(id: string): this;

    /**
     * Set the path to where command line history is persisted.
     * Must be called before vorpal.history
     * @param path
     * @return {this}
     * @api public
     */
    historyStoragePath(path: string): this;

    /**
     * Attaches the TTY's CLI prompt to that given instance of Vorpal.
     * As a note, multiple instances of Vorpal can run in the same Node
     * instance. However, only one can be 'attached' to your TTY. The last
     * instance given the show() command will be attached, and the previously
     * shown instances will detach.
     *
     * @returns {this}
     * @api public
     */
    show(): this;

    /**
     * Disables the vorpal prompt on the
     * local terminal.
     *
     * @return {this}
     * @api public
     */
    hide(): this;

    /**
     * For use in vorpal API commands, sends
     * a prompt command downstream to the local
     * terminal. Executes a prompt and returns
     * the response upstream to the API command.
     *
     * @param {Object} options
     * @param {Function} userCallback
     * @return {Vorpal}
     * @api public
     */
    prompt(): this;


    /**
     * Executes a vorpal API command and
     * returns the response either through a
     * callback or Promise in the absence
     * of a callback.
     *
     * A little black magic here - because
     * we sometimes have to send commands 10
     * miles upstream through 80 other instances
     * of vorpal and we aren't going to send
     * the callback / promise with us on that
     * trip, we store the command, callback,
     * resolve and reject objects (as they apply)
     * in a local vorpal._command variable.
     *
     * When the command eventually comes back
     * downstream, we dig up the callbacks and
     * finally resolve or reject the promise, etc.
     *
     * Lastly, to add some more complexity, we throw
     * command and callbacks into a queue that will
     * be unearthed and sent in due time.
     *
     * @param {String} command
     * @param {any} args
     * @param {Function} cb
     * @return {Promise or Vorpal}
     * @api public
     */
    exec<T>(
        command: string, args: {[k: string]: any, sessionId?: string},
        cb?: Vorpal.CallbackFunction<T>): PromiseLike<T>;

    exec<T>(command: string, cb?: Vorpal.CallbackFunction<T>): PromiseLike<T>;


    /**
     * Executes a Vorpal command in sync.
     *
     * @param {String} cmd
     * @param {Object} args
     * @return {*} stdout
     * @api public
     */
    execSync<T>(
        command: string, options: {[k: string]: any, sessionId?: string}): T;

    /**
     * Registers a custom handler for SIGINT.
     * Vorpal exits with 0 by default
     * on a sigint.
     *
     * @param {Function} fn
     * @return {Vorpal}
     * @api public
     */
    sigint(sigint: () => void): void;

    /**
     * Returns a given command by its name. This is used instead of
     * `vorpal.command()` as `.command` will overwrite a given command. If
     * command is not found, `undefined` is returned.
     *
     * @param {string} name
     * @returns {Command}
     */
    find(name: string): Vorpal.Command;


    /**
     * Registers custom help.
     *
     * @param {Function} fn
     * @return {Vorpal}
     * @api public
     */
    help(strBuilder: (cmd: string) => string): void;


    /**
     * Returns session by id.
     *
     * @param {Integer} id
     * @return {Session}
     * @api public
     */
    getSessionById(id: string): Vorpal.Session;
  }

  /*~ If you want to expose types from your module as well, you can
   *~ place them in this block.
   */
  namespace Vorpal {

    type CallbackFunction<T> = (err?: string|Error, data?: T) => void;

    interface TypesDefinition {
      string?: string[];
      number?: string[];
      // TODO: Check other types.
    }

    interface CommandData {
      command: string;
      description: string;
      options: Array<string>|Array<Array<string>>;
      action: CommandActionFn<any>;
    }

    type CommandOptions = Partial<{mode: boolean, 'catch': boolean}>;


    interface Session {}

    export interface UiInstance extends EventEmitter {
      redraw: RedrawMethod;

      delimiter(text?: string): this;

      input(text?: string): this;

      imprint(): this;

      submit(text: string): this;

      cancel(): this;
    }

    export function RedrawMethod(...texts: string[]): void;

    export interface RedrawMethod {
      clear(): void;

      done(): void;
    }

    export interface Command {
      description(description: string): this;

      delimiter(str: string): this;

      init(initFn: (args: any, callback: CallbackFunction<void>) => void): this;

      alias(name: string): this;

      alias(...names: string[]): this;

      parse(parseFn: CommandParseFn): this;

      option(flag: string, description: string, autocomplete: string[]):
          this;  // TODO: Check autocomplete types.
      types(types: TypesDefinition): this;

      hidden(): this;  // TODO: Check return type.
      remove(): this;  // TODO: Check return type.

      /**
       * Adds a custom handling for the --help flag.
       *
       * @param {Function} helpFn
       * @return {Command}
       * @api public
       */
      help(helpFn: () => void): this;  // TODO: Check args type.
      validate(validateFn: CommandValidateFn): this;

      autocomplete(choices: string[]): this;

      autocomplete(choices: {}): this;  // TODO: Revisit this.
      autocomplete<T>(choicesFn: CommandAutocompleteFn<T>):
          this;  // TODO: Revisit this.
      action<T>(actionFn: CommandActionFn<T>): this;
    }

    export interface Args {
      options: {[name: string]: string;}, [name: string]: string|object;
    }
    export interface CommandInstance {
      log(message?: any, ...optionalParams: any[]): void;
      prompt: Inquirer['prompt'];
      delimiter: Vorpal['delimiter'];
      args: Args;
      callback: any;
      command: any;
      commandObject: Command;
      commandWrapper: {
        args: Args; callback: CallbackFunction<any>; command: string;
        commandObject: Command;
        commandInstance: CommandInstance;
        fn: CommandActionFn<any>;
        pipes: Array<any>;
        session: Session;
        validate: any;
        downstream: any;
      };
      parent: Vorpal;
      session: Session;
    }

    type CommandParseFn = (command: string, args: Args) =>
        string;  // TODO: Check args type.
    type CommandValidateFn = (this: CommandInstance, args: Args) =>
        boolean|string;  // TODO: Check args type.
    type CommandAutocompleteFn<T> =
        (text: string, iteration: number,
         cb: CallbackFunction<T>) => void|PromiseLike<T>;
    type CommandActionFn<T> =
        (this: CommandInstance, args: Args,
         cb: CallbackFunction<T>) => void|PromiseLike<T>
  }

  export = Vorpal;
}
