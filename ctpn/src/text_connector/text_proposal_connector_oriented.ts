import * as tf from '@tensorflow/tfjs-core';
import {TextProposalGraphBuilder} from './TextProposalGraphBuilder';
import numeric from 'numeric';

class Poly1d{
    argA: tf.Tensor;
    argB: tf.Tensor;
    constructor(args: tf.Tensor) {
        this.argA = tf.gatherND(args,[0]);
        this.argB = tf.gatherND(args,[1]);
    }
    solve(x: number){
        // return this.argA * x + this.argB;
        return tf.add(tf.mul(this.argA,x), this.argB);
    }
    get equation(){
        return `${this.argA.arraySync()} x + ${this.argB.arraySync()}`;
    }
}

function polyfit<T extends tf.Tensor>(_x: T, _y: T, order: number){
    const xArray = _x.arraySync() as number[];
    const yArray = _y.arraySync() as number[];
    if (xArray.length <= order) {
        console.warn('Warning: Polyfit may be poorly conditioned.');
    }
    const xMatrix = [];
    const yMatrix = yArray;//numeric.transpose([yArray])

    for (let i = 0; i < xArray.length; i++) {
        const temp = [];
        for (let j = 0; j <= order; j++) {
            temp.push(Math.pow(xArray[i], j));
        }
        xMatrix.push(temp);
    }
    const xMatrixT = numeric.transpose(xMatrix);
    const dot1 = numeric.dot(xMatrixT, xMatrix);
    const dot2 = numeric.dot(xMatrixT, yMatrix);
    const dotInv = numeric.inv(dot1 as number[][]);
return tf.reverse(tf.tensor(numeric.dot(dotInv, dot2)));
}

export class TextProposalConnectorOriented{
    graphBuilder: TextProposalGraphBuilder;
    constructor() {
        this.graphBuilder = new TextProposalGraphBuilder();
    }

    async group_text_proposals<T extends tf.Tensor>(
        textProposals: T,
        scores: T,
        imSize: number[]
    ){
    const graph = this.graphBuilder.build_graph(textProposals, scores, imSize);
    return graph.sub_graphs_connected();
}

fit_y<T extends tf.Tensor>(X: T, Y: T, x1: T, x2: T){
    if( tf.equal(tf.gather(X,0), tf.div(tf.sum(X),X.shape[0])).arraySync()){
        return [tf.gatherND(Y,[0]), tf.gatherND(Y,[0])];
    }
    const p = new Poly1d(polyfit(X, Y, 1));
    return [
        p.solve(x1.arraySync() as number), p.solve(x2.arraySync() as number)
    ];
}

async get_text_lines<T extends tf.Tensor>(
    textProposals: T,
    scores: T,
    imSize: number[]
){
    const tpGroups = await this.group_text_proposals(
        textProposals, scores, imSize
    );

    const textLines = tf.buffer( [tpGroups.length, 8]);

    tpGroups.forEach((tpIndices,index)=>{
        const textLineBoxes = tf.gather(textProposals, tpIndices);
        const X = tf.div(
            tf.add(
                tf.reshape(
                    tf.slice(textLineBoxes,[0,0], [textLineBoxes.shape[0],1]),
                    [textLineBoxes.shape[0]]),
                tf.reshape(
                    tf.slice(textLineBoxes,[0,2],
                        [textLineBoxes.shape[0],1]),[textLineBoxes.shape[0]]) ),
            2);

        const Y = tf.div(
            tf.add(
                tf.reshape(
                    tf.slice(textLineBoxes,[0,1], [textLineBoxes.shape[0],1]),
                    [textLineBoxes.shape[0]]),
                tf.reshape(
                    tf.slice(textLineBoxes,[0,3],
                        [textLineBoxes.shape[0],1]),[textLineBoxes.shape[0]]) ),
            2);

        const z1 = polyfit(X, Y, 1);

        const x0 = tf.min(
            tf.reshape(
                tf.slice(textLineBoxes,[0,0], [textLineBoxes.shape[0],1]),
                [textLineBoxes.shape[0]]
            )
        );

        const x1 = tf.max(
            tf.reshape(
                tf.slice(
                    textLineBoxes,[0,2], [textLineBoxes.shape[0],1]),
                [textLineBoxes.shape[0]]
            )
        );

        const offset = tf.mul(
            tf.sub(
                tf.gatherND(textLineBoxes, [0,2]),
                tf.gatherND(textLineBoxes, [0,0])
            ),
            0.5);

        const [ltY, rtY] = this.fit_y(
            tf.reshape(
                tf.slice(
                    textLineBoxes,[0,0], [textLineBoxes.shape[0],1]
                ),[textLineBoxes.shape[0]]),
            tf.reshape(
                tf.slice(
                    textLineBoxes,[0,1], [textLineBoxes.shape[0],1]),
                [textLineBoxes.shape[0]]),
            tf.add(x0,offset), tf.sub(x1,offset));

        const [lbY, rbY] = this.fit_y(
            tf.reshape(
                tf.slice(textLineBoxes,[0,0], [textLineBoxes.shape[0],1]),
                [textLineBoxes.shape[0]]),
            tf.reshape(
                tf.slice(
                textLineBoxes,[0,3], [textLineBoxes.shape[0],1]),
                [textLineBoxes.shape[0]]),
            tf.add(x0,offset), tf.sub(x1,offset));

        const score = tf.div(
            tf.sum(tf.gather(scores,tpIndices)),
            tpIndices.length
        );

        textLines.set(x0.arraySync() as number, index, 0);
        textLines.set(tf.minimum(ltY, rtY).arraySync() as number, index, 1);
        textLines.set(x1.arraySync() as number, index, 2);
        textLines.set(tf.maximum(lbY, rbY).arraySync() as number, index, 3);
        textLines.set(score.arraySync() as number, index, 4);
        textLines.set(tf.gather(z1,[0]).arraySync() as number, index, 5);
        textLines.set(tf.gather(z1,[1]).arraySync() as number, index, 6);
        const height = tf.mean(
            tf.sub(
                tf.reshape(
                    tf.slice(textLineBoxes,[0,3], [textLineBoxes.shape[0],1]),
                    [textLineBoxes.shape[0]]),
                tf.reshape(tf.slice(textLineBoxes,[0,1],
                    [textLineBoxes.shape[0],1]),[textLineBoxes.shape[0]])) );

        textLines.set(tf.add(height,2.5).arraySync() as number, index, 7);

    });
    const _textLines = textLines.toTensor();
    const textRecs = tf.buffer( [textLines.shape[0], 9] );
    let index = 0;
    for(let i = 0; i< textLines.shape[0]; i++){

        const b1 = tf.sub(
            tf.gatherND(_textLines,[i, 6]),
            tf.div(tf.gatherND(_textLines,[i, 7]), 2)
        );
        const b2 = tf.add(
            tf.gatherND(_textLines,[i, 6]),
            tf.div(tf.gatherND(_textLines,[i, 7]), 2)
        );
        let x1 = tf.gatherND(_textLines,[i, 0]);
        let y1 = tf.add(
            tf.mul(
                tf.gatherND(_textLines,[i, 5]),
                tf.gatherND(_textLines,[i, 0])),
            b1);
        let x2 = tf.gatherND(_textLines,[i, 2]);
        let y2 = tf.add(
            tf.mul(tf.gatherND(_textLines,[i,5]),tf.gatherND(_textLines,[i,2])),
            b1);
        let x3 = tf.gatherND(_textLines,[i, 0]);
        let y3 = tf.add(
            tf.mul(
                tf.gatherND(_textLines,[i, 5]),
                tf.gatherND(_textLines,[i, 0])),
            b2);
        let x4 = tf.gatherND(_textLines,[i, 2]);
        let y4 = tf.add(
            tf.mul(
                tf.gatherND(_textLines,[i, 5]),
                tf.gatherND(_textLines,[i, 2])),
            b2);
        const disX = tf.sub(x2,x1);
        const disY = tf.sub(y2,y1);
        const width = tf.sqrt( tf.add (tf.mul(disX,disX), tf.mul(disY,disY)) );
        const fTmp0 = tf.sub(y3,y1);
        const fTmp1 = tf.div(tf.mul(fTmp0,disY),width);
        const x = tf.abs(tf.div(tf.mul(fTmp1,disX),width));
        const y = tf.abs(tf.div(tf.mul(fTmp1,disY),width) );

        if (tf.less(tf.gatherND(_textLines,[i, 5]),0).arraySync()){
            x1 = tf.sub(x1,x);
            y1 = tf.add(y1,y);
            x4 = tf.add(x4,x);
            y4 = tf.sub(y4,y);
        }else{
            x2 = tf.add(x2,x);
            y2 = tf.add(y2,y);
            x3 = tf.sub(x3,x);
            y3 = tf.sub(y3,y);
        }
        textRecs.set(x1.arraySync() as number, index, 0);
        textRecs.set(y1.arraySync() as number, index, 1);
        textRecs.set(x2.arraySync() as number, index, 2);
        textRecs.set(y2.arraySync() as number, index, 3);
        textRecs.set(x3.arraySync() as number, index, 4);
        textRecs.set(y3.arraySync() as number, index, 5);
        textRecs.set(x4.arraySync() as number, index, 6);
        textRecs.set(y4.arraySync() as number, index, 7);
        textRecs.set(
            tf.gatherND(_textLines,[i, 4])
                .arraySync() as number, index, 8);
        index+=1;

    }
    return textRecs.toTensor();
}

}
