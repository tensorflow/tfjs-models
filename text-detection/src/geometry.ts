export type Vector = [number, number];
export class Point {
  constructor(x: number, y: number) {
    this.x = x;
    this.y = y;
  }
  readonly x: number;
  readonly y: number;
  public add(otherPoint: Point): Point {
    return new Point(this.x + otherPoint.x, this.y + otherPoint.y);
  }
  public sub(otherPoint: Point): Point {
    return new Point(this.x - otherPoint.x, this.y - otherPoint.y);
  }
  public norm(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }
}
