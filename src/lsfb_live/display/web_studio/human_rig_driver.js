/**
 * Pilote cinématique par Quaternions pour un modèle humain 3D skinné Mixamo (67 os).
 *
 * Aligne hiérarchiquement les bras (épaule -> coude -> poignet), le repère orthonormé 3D
 * complet des paumes (orientation + pronation/supination) et les 30 os phalangiens.
 */
import * as THREE from "three";

const FINGER_BONE_CHAINS = [
  { prefix: "Thumb", indices: [1, 2, 3, 4] },
  { prefix: "Index", indices: [5, 6, 7, 8] },
  { prefix: "Middle", indices: [9, 10, 11, 12] },
  { prefix: "Ring", indices: [13, 14, 15, 16] },
  { prefix: "Pinky", indices: [17, 18, 19, 20] },
];

const _vA = new THREE.Vector3();
const _vB = new THREE.Vector3();
const _currDir = new THREE.Vector3();
const _targetDir = new THREE.Vector3();
const _qParent = new THREE.Quaternion();
const _qWorld = new THREE.Quaternion();
const _qDelta = new THREE.Quaternion();
const _qTargetLocal = new THREE.Quaternion();
const _mCurr = new THREE.Matrix4();
const _mTgt = new THREE.Matrix4();
const _mDelta = new THREE.Matrix4();

function toThreeVec(arr) {
  // Transformation (x, y, z) -> (x, -y, -z) de déterminant +1 (préserve la chiralité 3D)
  return new THREE.Vector3(arr[0], -arr[1], -arr[2]);
}

function buildOrthonormalBasis(origin, forwardPt, indexPt, pinkyPt, matOut) {
  const fwd = new THREE.Vector3().subVectors(forwardPt, origin).normalize();
  const toIdx = new THREE.Vector3().subVectors(indexPt, origin);
  const toPnk = new THREE.Vector3().subVectors(pinkyPt, origin);
  const side = new THREE.Vector3().subVectors(toIdx, toPnk).normalize();
  const normal = new THREE.Vector3().crossVectors(fwd, side).normalize();
  if (normal.lengthSq() < 1e-4) return false;
  const orthoSide = new THREE.Vector3().crossVectors(fwd, normal).normalize();
  matOut.makeBasis(orthoSide, fwd, normal);
  return true;
}

export class MixamoHumanRigDriver {
  constructor(rootObject) {
    this.root = rootObject;
    this.bones = {};
    this.restQuats = {};

    rootObject.traverse((obj) => {
      if (obj.isBone) {
        this.bones[obj.name] = obj;
        this.restQuats[obj.name] = obj.quaternion.clone();
      }
      if (obj.isSkinnedMesh) {
        obj.castShadow = true;
        obj.receiveShadow = true;
        obj.frustumCulled = false;
      }
    });

    // Stylisation technique sobre du mannequin biomécanique (ElmasCore)
    rootObject.traverse((obj) => {
      if (!obj.isMesh || !obj.material) return;
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach((m) => {
        if (m.name && m.name.toLowerCase().includes("joint")) {
          m.color.setHex(0xd97706);
          m.roughness = 0.35;
          m.metalness = 0.45;
        } else {
          m.color.setHex(0xcbd5e1);
          m.roughness = 0.48;
          m.metalness = 0.12;
        }
      });
    });
  }

  _resetUpperBodyToRest() {
    for (const [name, q] of Object.entries(this.restQuats)) {
      if (!name.includes("Leg") && !name.includes("Foot") && !name.includes("Toe")) {
        this.bones[name].quaternion.copy(q);
      }
    }
    this.root.updateMatrixWorld(true);
  }

  _aimBone(boneName, childName, fromPt, toPt, smooth = 0.55) {
    const bone = this.bones[boneName];
    const child = this.bones[childName];
    if (!bone || !child) return;

    _targetDir.subVectors(toPt, fromPt);
    if (_targetDir.lengthSq() < 1e-6) return;
    _targetDir.normalize();

    bone.updateWorldMatrix(true, false);
    child.updateWorldMatrix(true, false);
    bone.getWorldPosition(_vA);
    child.getWorldPosition(_vB);
    _currDir.subVectors(_vB, _vA);
    if (_currDir.lengthSq() < 1e-8) return;
    _currDir.normalize();

    _qDelta.setFromUnitVectors(_currDir, _targetDir);
    bone.parent.getWorldQuaternion(_qParent);
    bone.getWorldQuaternion(_qWorld);

    _qTargetLocal.copy(_qParent).invert().multiply(_qDelta).multiply(_qWorld);
    if (smooth < 1.0 && bone.userData.prevQuat) {
      bone.userData.prevQuat.slerp(_qTargetLocal, smooth);
      bone.quaternion.copy(bone.userData.prevQuat);
    } else {
      bone.quaternion.copy(_qTargetLocal);
      bone.userData.prevQuat = _qTargetLocal.clone();
    }
    bone.updateMatrixWorld(true);
  }

  _orientHandAndFingers(sidePrefix, handPts, isActive) {
    const handBone = this.bones[`mixamorig:${sidePrefix}Hand`];
    const idx1 = this.bones[`mixamorig:${sidePrefix}HandIndex1`];
    const mid1 = this.bones[`mixamorig:${sidePrefix}HandMiddle1`];
    const pnk1 = this.bones[`mixamorig:${sidePrefix}HandPinky1`];
    if (!handBone || !idx1 || !mid1 || !pnk1) return;

    handBone.updateWorldMatrix(true, true);
    const pHand = handBone.getWorldPosition(new THREE.Vector3());
    const pIdx = idx1.getWorldPosition(new THREE.Vector3());
    const pMid = mid1.getWorldPosition(new THREE.Vector3());
    const pPnk = pnk1.getWorldPosition(new THREE.Vector3());

    const okCurr = buildOrthonormalBasis(pHand, pMid, pIdx, pPnk, _mCurr);
    const okTgt = buildOrthonormalBasis(handPts[0], handPts[9], handPts[5], handPts[17], _mTgt);

    if (okCurr && okTgt) {
      _mDelta.copy(_mTgt).multiply(_mCurr.clone().transpose());
      _qDelta.setFromRotationMatrix(_mDelta);
      handBone.parent.getWorldQuaternion(_qParent);
      handBone.getWorldQuaternion(_qWorld);
      _qTargetLocal.copy(_qParent).invert().multiply(_qDelta).multiply(_qWorld);

      if (handBone.userData.prevQuat) {
        handBone.userData.prevQuat.slerp(_qTargetLocal, 0.55);
        handBone.quaternion.copy(handBone.userData.prevQuat);
      } else {
        handBone.quaternion.copy(_qTargetLocal);
        handBone.userData.prevQuat = _qTargetLocal.clone();
      }
      handBone.updateMatrixWorld(true);
    }

    if (!isActive) return;

    // Alignement hiérarchique des 15 phalanges de la main (3 os par doigt)
    for (const finger of FINGER_BONE_CHAINS) {
      const [j0, j1, j2, j3] = finger.indices;
      this._aimBone(
        `mixamorig:${sidePrefix}Hand${finger.prefix}1`,
        `mixamorig:${sidePrefix}Hand${finger.prefix}2`,
        handPts[j0], handPts[j1], 0.65
      );
      this._aimBone(
        `mixamorig:${sidePrefix}Hand${finger.prefix}2`,
        `mixamorig:${sidePrefix}Hand${finger.prefix}3`,
        handPts[j1], handPts[j2], 0.65
      );
      this._aimBone(
        `mixamorig:${sidePrefix}Hand${finger.prefix}3`,
        `mixamorig:${sidePrefix}Hand${finger.prefix}4`,
        handPts[j2], handPts[j3], 0.65
      );
    }
  }

  applyRigState(rig) {
    this._resetUpperBodyToRest();

    // Conversion des cibles 3D en espace Three.js (+Y haut, +Z face caméra)
    const ls = toThreeVec(rig.left_shoulder);
    const rs = toThreeVec(rig.right_shoulder);
    const le = toThreeVec(rig.left_elbow);
    const re = toThreeVec(rig.right_elbow);
    const lw = toThreeVec(rig.left_wrist);
    const rw = toThreeVec(rig.right_wrist);
    const lhPts = rig.left_hand.map(toThreeVec);
    const rhPts = rig.right_hand.map(toThreeVec);

    // Micro-orientation naturelle de la tête vers l'action
    const headBone = this.bones["mixamorig:Head"];
    if (headBone) {
      const yaw = THREE.MathUtils.clamp(rig.head_center[0] * 0.45, -0.22, 0.22);
      const pitch = THREE.MathUtils.clamp((rig.head_center[1] + 0.62) * 0.35, -0.15, 0.15);
      headBone.rotation.set(pitch, yaw, 0);
      headBone.updateMatrixWorld(true);
    }

    // Miroir exact (déterminant +1) :
    // - Côté gauche de l'écran (rig.left_*, X < 0) -> Bras Droit anatomique du personnage (mixamorig:Right*)
    // - Côté droit de l'écran (rig.right_*, X > 0) -> Bras Gauche anatomique du personnage (mixamorig:Left*)
    this._aimBone("mixamorig:RightArm", "mixamorig:RightForeArm", ls, le, 0.55);
    this._aimBone("mixamorig:RightForeArm", "mixamorig:RightHand", le, lw, 0.55);
    this._orientHandAndFingers("Right", lhPts, rig.left_hand_active !== false);

    this._aimBone("mixamorig:LeftArm", "mixamorig:LeftForeArm", rs, re, 0.55);
    this._aimBone("mixamorig:LeftForeArm", "mixamorig:LeftHand", re, rw, 0.55);
    this._orientHandAndFingers("Left", rhPts, rig.right_hand_active !== false);
  }
}
