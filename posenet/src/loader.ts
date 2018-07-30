// async function loadRemoteProtoFile(modelUrl: string):
//     Promise<tensorflow.GraphDef> {
//   try {
//     const response = await fetch(this.modelUrl, this.requestOption);
//     return tensorflow.GraphDef.decode(
//         new Uint8Array(await response.arrayBuffer()));
//   } catch (error) {
//     throw new Error(`${this.modelUrl} not found. ${error}`);
//   }
// }
