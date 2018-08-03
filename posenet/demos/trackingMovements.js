import {drawPoint, drawRect} from './demo_util';

export const videoDimensions = () => {
  const video = document.getElementById('video');
  return ({height: video.height, width: video.width});
};

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
export const trackFeetTogether = (ctx, keypoints) => {
  const feetDist = keypoints[15].position.x - keypoints[16].position.x;
  const width = videoDimensions().width;
  const startRect = (width/2)-(width/8);

  if (Math.abs(feetDist) < 25) {
    drawRect(ctx, width/4, 500, 'rgba(51, 225, 91, 0.25)', startRect, 0);
  }
};

// track feet left or right side of screen
export function trackFeet(ctx, keypoints) {
  const rightFootX = keypoints[15].position.x;
  const leftFootX = keypoints[16].position.x;
  const feetDist = rightFootX - leftFootX;
  const width = videoDimensions().width;
  const midpoint = width/2;


  if (Math.abs(feetDist) >= 25) {

    if (leftFootX < (midpoint-width/8)) {
    drawRect(ctx, midpoint-width/8, 500, 'rgba(225, 161, 51, 0.52)', 0, 0);
    }

    if (rightFootX > (midpoint+width/8)) {
      drawRect(ctx, midpoint-25, 500, 'rgba(51, 225, 196, 0.52)', midpoint+width/8, 0);
      };
  }
}

// track jump (both feet above a certain height)
export const trackJump = (ctx, keypoints) => {
  const rightFootY = keypoints[15].position.y;
  const leftFootY = keypoints[16].position.y;
  const height = videoDimensions().height;
  const width = videoDimensions().width;
  const barHeight = height-(height/8);

  console.log('rightFoot, leftFoot:', rightFootY, leftFootY);
  console.log('barHeight', barHeight);

  if ((rightFootY < barHeight) && (leftFootY < barHeight)) {
    drawRect(
      ctx, width, barHeight,
      'rgba(215, 180, 54, 0.93)',
      0, barHeight
    );
  }
};


