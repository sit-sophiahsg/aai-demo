// =============================================================================
//  modes/sign.js: SIGN mode (ASL fingerspelling), built in Phase 3
// =============================================================================
//  For now this is a placeholder that implements the mode interface (see
//  modes/draw.js), so mode switching and particle trails can be tested.

let app = null;

export const signMode = {
  id: "sign",
  name: "Sign",
  init(ctx) { app = ctx; },
  enter() {},
  exit() {},
  update() {},
  render(ctx) {
    app.drawCenterText("Sign Language", "coming in Phase 3");
  },
  handLost() {},
  trailColor(hand) { return app.handColors[hand.label]; },
  labelSuffix() { return ""; },
  badge() { return { text: "SIGN", tone: "HOVER" }; },
  onKey() { return false; },
};
