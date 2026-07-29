(() => {
    "use strict";

    const STATION_MARKS = {
        acetaminophen: "对乙",
        ibuprofen: "布洛",
        ors: "补液",
        pharmacist: "药师",
        emergency: "急诊",
        quarantine: "隔离",
    };

    const TONES = {
        coral: 0xef716b,
        blue: 0x4c87dd,
        cyan: 0x40b9d0,
        violet: 0x9478c8,
        red: 0xd94d55,
        gold: 0xdca43e,
    };

    class VitalHospitalGame {
        constructor({ parent, actions, onAction }) {
            if (!window.Phaser) throw new Error("Phaser 运行库不可用");
            this.parent = parent;
            this.actions = actions || [];
            this.onAction = onAction;
            this.game = null;
            this.scene = null;
            this.session = null;
            this.caseData = null;
            this.lockedState = true;
            this.externalMove = { left: false, right: false, up: false, down: false };
            this.targetStation = null;
            this.sceneReady = null;
            this.bindTouchControls();
        }

        async start(session) {
            this.session = session;
            this.lockedState = false;
            if (!this.game) {
                await this.createGame();
            } else {
                this.scene.readyForCases = false;
                this.scene.scene.restart({ actions: session.actions || this.actions });
                await new Promise((resolve) => {
                    const wait = () => {
                        if (this.scene && this.scene.readyForCases) resolve();
                        else window.setTimeout(wait, 20);
                    };
                    wait();
                });
            }
            this.resume();
        }

        createGame() {
            const owner = this;
            this.sceneReady = new Promise((resolve) => {
                class HospitalScene extends Phaser.Scene {
                    constructor() {
                        super("VitalHospitalShift");
                    }

                    create(data) {
                        owner.scene = this;
                        this.owner = owner;
                        this.readyForCases = false;
                        this.actions = data.actions || owner.actions;
                        this.stationObjects = [];
                        this.floorGraphics = this.add.graphics().setDepth(0);
                        this.wallGraphics = this.add.graphics().setDepth(1);
                        this.decorGraphics = this.add.graphics().setDepth(2);
                        this.createStations();
                        this.createPatient();
                        this.createPlayer();
                        this.createPulseMonitor();
                        this.keys = this.input.keyboard.addKeys({
                            up: Phaser.Input.Keyboard.KeyCodes.W,
                            down: Phaser.Input.Keyboard.KeyCodes.S,
                            left: Phaser.Input.Keyboard.KeyCodes.A,
                            right: Phaser.Input.Keyboard.KeyCodes.D,
                            action: Phaser.Input.Keyboard.KeyCodes.E,
                            arrowUp: Phaser.Input.Keyboard.KeyCodes.UP,
                            arrowDown: Phaser.Input.Keyboard.KeyCodes.DOWN,
                            arrowLeft: Phaser.Input.Keyboard.KeyCodes.LEFT,
                            arrowRight: Phaser.Input.Keyboard.KeyCodes.RIGHT,
                            space: Phaser.Input.Keyboard.KeyCodes.SPACE,
                        });
                        this.actionWasDown = false;
                        this.scale.on("resize", this.layout, this);
                        this.layout();
                        this.readyForCases = true;
                        resolve();
                    }

                    createStations() {
                        this.actions.forEach((action, index) => {
                            const container = this.add.container(0, 0).setDepth(4);
                            const shadow = this.add.graphics();
                            shadow.fillStyle(0x061116, 0.28);
                            shadow.fillRoundedRect(-62, -22, 132, 70, 7);
                            const base = this.add.graphics();
                            base.fillStyle(0x29424b, 1);
                            base.fillRoundedRect(-66, -31, 132, 68, 7);
                            base.fillStyle(0x192c34, 1);
                            base.fillRoundedRect(-66, 22, 132, 21, { bl: 7, br: 7, tl: 0, tr: 0 });
                            base.fillStyle(TONES[action.tone] || 0x3fcbb3, 1);
                            base.fillRect(-66, -31, 132, 5);
                            base.lineStyle(1, 0x8fa8af, 0.22);
                            base.strokeRoundedRect(-66, -31, 132, 68, 7);
                            const markPlate = this.add.graphics();
                            markPlate.fillStyle(0x0d2028, 0.92);
                            markPlate.fillRoundedRect(-47, -17, 40, 34, 5);
                            const mark = this.add.text(-27, 0, STATION_MARKS[action.key] || "核对", {
                                color: "#ffffff",
                                fontFamily: "Microsoft YaHei",
                                fontSize: "11px",
                                fontStyle: "bold",
                            }).setOrigin(0.5);
                            const label = this.add.text(15, -8, action.label, {
                                color: "#f1f6f7",
                                fontFamily: "Microsoft YaHei",
                                fontSize: "10px",
                                fontStyle: "bold",
                                align: "left",
                                wordWrap: { width: 72 },
                            }).setOrigin(0.5);
                            const sub = this.add.text(15, 10, action.short, {
                                color: "#78939d",
                                fontFamily: "Microsoft YaHei",
                                fontSize: "7px",
                            }).setOrigin(0.5);
                            const signal = this.add.graphics();
                            signal.lineStyle(2, TONES[action.tone] || 0x42d7bd, 0.9);
                            signal.strokeRoundedRect(-72, -37, 144, 82, 9);
                            signal.setAlpha(0);
                            container.add([shadow, base, markPlate, mark, label, sub, signal]);
                            container.setSize(144, 84);
                            container.setInteractive({ useHandCursor: true });
                            container.on("pointerdown", () => this.moveToStation(action.key));
                            container.on("pointerover", () => signal.setAlpha(0.7));
                            container.on("pointerout", () => {
                                if (this.nearestStation?.action.key !== action.key) signal.setAlpha(0);
                            });
                            this.stationObjects.push({ action, container, signal, index });
                        });
                    }

                    createPatient() {
                        this.patient = this.add.container(0, 0).setDepth(5);
                        const shadow = this.add.ellipse(0, 31, 48, 16, 0x061116, 0.28);
                        const body = this.add.graphics();
                        body.fillStyle(0x395d68, 1);
                        body.fillRoundedRect(-21, -3, 42, 45, 12);
                        body.fillStyle(0xf0b84c, 1);
                        body.fillRect(-21, 31, 42, 5);
                        const head = this.add.circle(0, -17, 15, 0xd9a47e);
                        const hair = this.add.graphics();
                        hair.fillStyle(0x253139, 1);
                        hair.fillRoundedRect(-15, -30, 30, 12, 6);
                        const face = this.add.text(0, -15, "••", {
                            color: "#30393d", fontFamily: "Arial", fontSize: "8px",
                        }).setOrigin(0.5);
                        this.patientName = this.add.text(0, 56, "WAITING", {
                            backgroundColor: "#102229",
                            color: "#cbd9dc",
                            fontFamily: "Microsoft YaHei",
                            fontSize: "8px",
                            padding: { x: 7, y: 4 },
                        }).setOrigin(0.5);
                        this.patient.add([shadow, body, head, hair, face, this.patientName]);
                    }

                    createPlayer() {
                        this.player = this.add.container(0, 0).setDepth(8);
                        const shadow = this.add.ellipse(0, 28, 44, 15, 0x061116, 0.3);
                        const body = this.add.graphics();
                        body.fillStyle(0xe8f3f3, 1);
                        body.fillRoundedRect(-18, -5, 36, 41, 10);
                        body.fillStyle(0x32b9a2, 1);
                        body.fillRect(-18, 25, 36, 5);
                        const head = this.add.circle(0, -18, 14, 0xe3ad84);
                        const hair = this.add.graphics();
                        hair.fillStyle(0x172a32, 1);
                        hair.fillRoundedRect(-14, -30, 28, 11, 6);
                        const badge = this.add.graphics();
                        badge.fillStyle(0xef6f69, 1);
                        badge.fillRect(-3, 5, 6, 18);
                        badge.fillRect(-9, 11, 18, 6);
                        this.player.add([shadow, body, head, hair, badge]);
                        this.physics.add.existing(this.player);
                        this.player.body.setSize(34, 44);
                        this.player.body.setOffset(-17, -13);
                        this.player.body.setCollideWorldBounds(true);
                        this.player.body.setMaxVelocity(270, 270);
                    }

                    createPulseMonitor() {
                        this.monitor = this.add.container(0, 0).setDepth(3);
                        const body = this.add.graphics();
                        body.fillStyle(0x0a1a20, 1);
                        body.fillRoundedRect(-72, -32, 144, 64, 6);
                        body.lineStyle(1, 0x4bbda9, 0.35);
                        body.strokeRoundedRect(-72, -32, 144, 64, 6);
                        const title = this.add.text(-57, -22, "VITAL OPS", {
                            color: "#6be2cd", fontFamily: "Arial", fontSize: "7px", fontStyle: "bold",
                        });
                        this.pulse = this.add.graphics();
                        this.monitor.add([body, title, this.pulse]);
                    }

                    layout() {
                        const width = this.scale.width;
                        const height = this.scale.height;
                        this.physics.world.setBounds(20, 20, Math.max(40, width - 40), Math.max(40, height - 40));
                        this.drawRoom(width, height);
                        const marginX = Math.max(96, width * 0.1);
                        const topY = Math.max(104, height * 0.19);
                        const bottomY = Math.min(height - 92, height * 0.79);
                        const positions = [
                            [marginX + width * 0.12, topY],
                            [width / 2, topY],
                            [width - marginX - width * 0.12, topY],
                            [marginX + width * 0.12, bottomY],
                            [width / 2, bottomY],
                            [width - marginX - width * 0.12, bottomY],
                        ];
                        this.stationObjects.forEach((station, index) => {
                            const position = positions[index % positions.length];
                            station.container.setPosition(position[0], position[1]);
                        });
                        this.patient.setPosition(Math.max(58, width * 0.055), height / 2);
                        this.monitor.setPosition(width - 92, height / 2);
                        if (!this.hasPlacedPlayer) {
                            this.player.setPosition(width / 2, height / 2);
                            this.hasPlacedPlayer = true;
                        }
                    }

                    drawRoom(width, height) {
                        const floor = this.floorGraphics;
                        floor.clear();
                        floor.fillStyle(0x173039, 1);
                        floor.fillRect(0, 0, width, height);
                        const tile = Math.max(38, Math.min(58, width / 20));
                        for (let y = 0; y < height + tile; y += tile) {
                            for (let x = 0; x < width + tile; x += tile) {
                                const offset = (Math.floor(y / tile) % 2) * tile * 0.5;
                                floor.fillStyle((Math.floor(x / tile) + Math.floor(y / tile)) % 2 ? 0x1b3740 : 0x1d3b44, 1);
                                floor.fillRect(x - offset, y, tile - 1, tile - 1);
                            }
                        }
                        floor.fillStyle(0x264a52, 0.72);
                        floor.fillRect(width * 0.2, height * 0.34, width * 0.6, height * 0.32);
                        floor.lineStyle(2, 0x53d8c0, 0.15);
                        floor.strokeRect(width * 0.2, height * 0.34, width * 0.6, height * 0.32);

                        const walls = this.wallGraphics;
                        walls.clear();
                        walls.fillStyle(0x0a1c23, 1);
                        walls.fillRect(0, 0, width, 20);
                        walls.fillRect(0, height - 20, width, 20);
                        walls.fillRect(0, 0, 20, height);
                        walls.fillRect(width - 20, 0, 20, height);
                        walls.fillStyle(0x44cdb5, 1);
                        walls.fillRect(0, 20, 5, height - 40);
                        walls.fillStyle(0xef716b, 1);
                        walls.fillRect(width - 5, 20, 5, height - 40);

                        const decor = this.decorGraphics;
                        decor.clear();
                        decor.fillStyle(0x10262e, 1);
                        decor.fillRoundedRect(30, height / 2 - 78, 82, 156, 7);
                        decor.fillStyle(0x2d5a60, 1);
                        decor.fillRect(44, height / 2 - 56, 54, 8);
                        decor.fillRect(44, height / 2 + 48, 54, 8);
                        decor.fillStyle(0x10262e, 1);
                        decor.fillRoundedRect(width - 164, height / 2 - 68, 134, 136, 7);
                    }

                    moveToStation(actionKey) {
                        if (this.owner.lockedState) return;
                        const station = this.stationObjects.find((item) => item.action.key === actionKey);
                        if (!station) return;
                        this.owner.targetStation = station;
                    }

                    interactNearest() {
                        if (this.owner.lockedState || !this.nearestStation) return;
                        const distance = Phaser.Math.Distance.Between(
                            this.player.x, this.player.y,
                            this.nearestStation.container.x, this.nearestStation.container.y,
                        );
                        if (distance <= 118) this.owner.onAction(this.nearestStation.action.key);
                    }

                    setCase(caseData, index) {
                        this.patientName.setText(caseData.patient);
                        const skin = [0xd9a47e, 0xe8bd98, 0xbe825d, 0xe1aa85][index % 4];
                        const head = this.patient.list[2];
                        head.setFillStyle(skin);
                        this.patient.setAlpha(0);
                        this.patient.x -= 24;
                        this.tweens.add({ targets: this.patient, alpha: 1, x: this.patient.x + 24, duration: 360, ease: "Cubic.Out" });
                        this.owner.lockedState = false;
                        this.owner.targetStation = null;
                    }

                    showFeedback(correct) {
                        const color = correct ? 0x51dec5 : 0xee716b;
                        for (let index = 0; index < 18; index += 1) {
                            const angle = index / 18 * Math.PI * 2;
                            const distance = 45 + (index % 5) * 16;
                            const particle = this.add.rectangle(this.player.x, this.player.y, 5, 5, color).setDepth(12).setRotation(angle);
                            this.tweens.add({
                                targets: particle,
                                x: this.player.x + Math.cos(angle) * distance,
                                y: this.player.y + Math.sin(angle) * distance,
                                alpha: 0,
                                scale: 0.2,
                                duration: 430 + (index % 4) * 45,
                                ease: "Cubic.Out",
                                onComplete: () => particle.destroy(),
                            });
                        }
                        this.cameras.main.flash(160, correct ? 45 : 180, correct ? 170 : 45, correct ? 145 : 45, false);
                    }

                    update(time) {
                        if (!this.player?.body) return;
                        const owner = this.owner;
                        const keyboard = this.keys;
                        const left = keyboard.left.isDown || keyboard.arrowLeft.isDown || owner.externalMove.left;
                        const right = keyboard.right.isDown || keyboard.arrowRight.isDown || owner.externalMove.right;
                        const up = keyboard.up.isDown || keyboard.arrowUp.isDown || owner.externalMove.up;
                        const down = keyboard.down.isDown || keyboard.arrowDown.isDown || owner.externalMove.down;
                        const manual = left || right || up || down;
                        if (manual) owner.targetStation = null;
                        this.player.body.setVelocity(0);
                        if (!owner.lockedState) {
                            if (manual) {
                                const vector = new Phaser.Math.Vector2((right ? 1 : 0) - (left ? 1 : 0), (down ? 1 : 0) - (up ? 1 : 0)).normalize().scale(235);
                                this.player.body.setVelocity(vector.x, vector.y);
                            } else if (owner.targetStation) {
                                const target = owner.targetStation.container;
                                const distance = Phaser.Math.Distance.Between(this.player.x, this.player.y, target.x, target.y);
                                if (distance > 84) {
                                    const vector = new Phaser.Math.Vector2(target.x - this.player.x, target.y - this.player.y).normalize().scale(220);
                                    this.player.body.setVelocity(vector.x, vector.y);
                                } else {
                                    owner.targetStation = null;
                                    this.nearestStation = this.stationObjects.find((item) => item.container === target);
                                    this.interactNearest();
                                }
                            }
                        }
                        const moving = this.player.body.velocity.lengthSq() > 0;
                        this.player.rotation = moving ? Math.sin(time * 0.015) * 0.018 : 0;

                        let nearest = null;
                        let nearestDistance = Infinity;
                        this.stationObjects.forEach((station) => {
                            const distance = Phaser.Math.Distance.Between(this.player.x, this.player.y, station.container.x, station.container.y);
                            if (distance < nearestDistance) {
                                nearest = station;
                                nearestDistance = distance;
                            }
                        });
                        this.nearestStation = nearestDistance <= 118 ? nearest : null;
                        this.stationObjects.forEach((station) => {
                            station.signal.setAlpha(this.nearestStation === station && !owner.lockedState ? 0.85 : 0);
                        });

                        const actionDown = Phaser.Input.Keyboard.JustDown(keyboard.action) || Phaser.Input.Keyboard.JustDown(keyboard.space);
                        if (actionDown) this.interactNearest();
                        this.drawPulse(time);
                    }

                    drawPulse(time) {
                        const pulse = this.pulse;
                        pulse.clear();
                        pulse.lineStyle(2, 0x50d9c0, 0.85);
                        const points = [];
                        for (let x = -56; x <= 57; x += 3) {
                            const cycle = ((x + time * 0.08) % 48 + 48) % 48;
                            const spike = cycle > 20 && cycle < 24 ? -17 + Math.abs(cycle - 22) * 8 : cycle >= 24 && cycle < 28 ? 11 - Math.abs(cycle - 26) * 5 : 0;
                            points.push(new Phaser.Math.Vector2(x, 8 + spike));
                        }
                        pulse.strokePoints(points, false, false);
                    }
                }

                const parent = document.getElementById(this.parent);
                const config = {
                    type: Phaser.AUTO,
                    parent: this.parent,
                    backgroundColor: "#173039",
                    width: Math.max(320, parent.clientWidth || 960),
                    height: Math.max(420, parent.clientHeight || 540),
                    transparent: false,
                    antialias: true,
                    render: { pixelArt: false, roundPixels: true },
                    scale: { mode: Phaser.Scale.RESIZE, autoCenter: Phaser.Scale.CENTER_BOTH },
                    physics: { default: "arcade", arcade: { debug: false } },
                    scene: HospitalScene,
                    input: { activePointers: 3 },
                    banner: false,
                };
                owner.game = new Phaser.Game(config);
                owner.game.events.once(Phaser.Core.Events.READY, () => {
                    if (owner.scene?.readyForCases) resolve();
                });
            });
            return this.sceneReady;
        }

        setCase(caseData, index) {
            this.caseData = caseData;
            this.lockedState = false;
            this.targetStation = null;
            if (this.scene?.readyForCases) this.scene.setCase(caseData, index);
        }

        lock() {
            this.lockedState = true;
            this.targetStation = null;
        }

        unlock() {
            this.lockedState = false;
        }

        feedback(correct) {
            if (this.scene?.readyForCases) this.scene.showFeedback(correct);
        }

        complete() {
            this.lockedState = true;
            this.targetStation = null;
            if (this.scene?.physics) this.scene.physics.pause();
        }

        pause() {
            if (this.scene?.scene?.isActive()) this.scene.scene.pause();
        }

        resume() {
            if (this.scene?.scene?.isPaused()) this.scene.scene.resume();
            if (this.scene?.physics?.world?.isPaused) this.scene.physics.resume();
        }

        resize() {
            if (this.game?.scale) {
                const parent = document.getElementById(this.parent);
                if (parent?.clientWidth && parent?.clientHeight) {
                    this.game.scale.resize(parent.clientWidth, parent.clientHeight);
                }
            }
        }

        bindTouchControls() {
            document.querySelectorAll("[data-game-move]").forEach((button) => {
                const direction = button.dataset.gameMove;
                const start = (event) => {
                    event.preventDefault();
                    this.externalMove[direction] = true;
                };
                const stop = (event) => {
                    event.preventDefault();
                    this.externalMove[direction] = false;
                };
                button.addEventListener("pointerdown", start);
                button.addEventListener("pointerup", stop);
                button.addEventListener("pointercancel", stop);
                button.addEventListener("pointerleave", stop);
            });
            const action = document.getElementById("gameTouchAction");
            if (action) {
                action.addEventListener("click", () => {
                    if (this.scene?.readyForCases) this.scene.interactNearest();
                });
            }
        }
    }

    window.VitalHospitalGame = VitalHospitalGame;
})();
