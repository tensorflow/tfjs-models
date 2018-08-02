import {drawPoint, drawRect} from './demo_util';

// track shoulders (which is higher)
export function trackShoulders(ctx, keypoints) {
  const shoulderDif = keypoints[9].position.y - keypoints[10].position.y;
  if (Math.abs(shoulderDif) > 15) {
    // console.log(
    //   'higher shoulder:',
    //   (shoulderDif > 0)
    //     ? 'left'
    //     : 'right');

    if (shoulderDif > 0) {
    drawPoint(ctx, 300, 100, 20, 'orange');
    } else if (shoulderDif < 0) {
    drawPoint(ctx, 300, 500, 20, 'orange');
    }
  }
}

// track hips
export function trackHips(ctx, keypoints) {
  const hipDif = keypoints[11].position.y - keypoints[12].position.y;
  const leftHipPastleftFoot = keypoints[11].position.x - keypoints[15].position.x;
  if ((leftHipPastleftFoot) > 5) {
    // console.log(
    //   'higher shoulder:',
    //   (shoulderDif > 0)
    //     ? 'left'
    //     : 'right');

    if (leftHipPastleftFoot > 0) {
    drawPoint(ctx, 400, 100, 20, 'red');
    } else if (leftHipPastleftFoot < 0) {
    drawPoint(ctx, 400, 500, 20, 'red');
    }
  }
}

// track feet (apart or together)
export function trackFeetTogether(ctx, keypoints) {
  const feetDist = keypoints[15].position.x - keypoints[16].position.x;
  if (Math.abs(feetDist) < 25) {
    console.log('feetDist:', feetDist);

    drawPoint(ctx, 400, 200, 20, 'green');
    drawRect(ctx, 300, 500, 'rgba(51, 225, 91, 0.15)');
  }
}
