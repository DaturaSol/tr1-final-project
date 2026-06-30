# Projeto Final de Teleinformática e Redes

>[!IMPORTANT]
> Nome: Gabriel M.S.O.  
> Matrícula: 190042656  
> 
> O uso de IA neste projeto será unicamente para correções gramaticais e pesquisa de funções. Por exemplo, aposto que você não sabia que o módulo `NumPy` possui a função `einsum`, que realiza a soma de Einstein, e que módulos como o `PyTorch`, que herdam do `NumPy`, também incluem a mesma função.
>
> Todos os códigos serão escritos e documentados em inglês. Gosto de manter o código em apenas uma linguagem; assim, tudo fica organizado e fácil de entender para qualquer pessoa.

## Como utilizar

### 1. Com VS Code e Dev Containers

Tenha o VS Code instalado, baixe a extensão dev-containers (`ms-azuretools.vscode-containers`) e tenha o [Docker Desktop instalado](https://docs.docker.com/desktop/setup/install/windows-install/). Reconstrua e reabra dentro do container, e então rode

```bash
uv run streamlit run app/app.py
```

### 2. Com Docker

Tenha apenas o Docker instalado

```bash
cd docker
docker compose up -d --build
docker compose exec dev uv sync
```

Em outro terminal
```bash
docker compose exec rx uv run python nodes/rx/main.py 
```

Em outro terminal
```bash
docker compose exec ch uv run python nodes/ch/main.py 
```

Em outro terminal
```bash
docker compose exec -e TR1_DIGITAL=manchester -e TR1_CARRIER=qpsk \
    tx uv run python nodes/tx/main.py "Ain batman" 
```

### 3. Sem Docker

Agora, para os rebeldes que não querem Docker, [instala o uv](https://docs.astral.sh/uv/getting-started/installation/)

e roda 
```bash 
uv sync
uv run python nodes/rx/main.py
```

Em outro terminal
```bash 
RX_HOST=127.0.0.1 uv run python nodes/ch/main.py
```

Em outro terminal 
```bash
CH_HOST=127.0.0.1 uv run python nodes/tx/main.py "Ain batman"
```


## Introdução

Desejamos simular a camada física de transmissão. Para isto dividimos este projeto em duas partes: containers completamente isolados rodando no Docker que se comunicam entre si, e uma simulação rápida em `app/app.py`.

Uma breve explicação de cada arquivo:

- `wire.py` conjunto comum de funções entre os containers para que eles se comuniquem.
- `utils.py` funções utilitárias que fizemos logo no início, usadas para converter de texto para bits e de bits para texto.
- `types.py` tipos de sinais, só possuímos bits e o sinal captado pelo fio.
- `protocol.py` protocolos utilizados logo no início para melhor organizar as saídas e entradas.
- `pipeline.py` como cada camada roda e a ordem em que os componentes funcionam.
- `factory.py` mapeia as configurações para suas implementações.
- `phy/` conteúdo da camada física.
- `link/` conteúdo da camada de enlace.

## phy

### channel

Implementação mais simples da camada física, que seria adicionar ruído.

```python
class GaussianChannel(Channel):
    """Adds Gaussian noise n(x, sigma) to each V/W sample of the signal."""

    def __init__(self, mean: float, std: float) -> None:
        """Store the noise distribution parameters.

        Args:
            mean: Mean (x) of the Gaussian noise, in the signal's units.
            std: Standard deviation (sigma) of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def transmit(self, signal: Signal) -> Signal:
        """Return ``signal`` with a Gaussian noise sample added per element.

        Args:
            signal: The transmitted samples, in V/W.

        Returns:
            ``signal + n`` where each ``n`` is drawn from ``N(mean, std)``.
            With ``std == 0`` the signal passes through unchanged.
        """
        noise = np.random.normal(
            loc=self.mean, scale=self.std, size=signal.shape
        )
        noisy_signal: Signal = signal + noise
        return noisy_signal
```

Simples, `noisy_signal: Signal = signal + noise`, se adiciona o ruído gerado com `np.random.normal` ao sinal resultante.

### baseband

```python
class BasebandModulator(Modulator):
    """Shared parameters for the baseband line codes below."""

    def __init__(self, amplitude: float, samples_per_symbol: int = 2) -> None:
        """Store the voltage level and per-bit sample count.

        Args:
            amplitude: Voltage level in volts; the sign convention is set by
                each concrete line code.
            samples_per_symbol: Discrete samples emitted per bit (>= 2).
        """
        self.amplitude = amplitude
        self.samples_per_symbol = samples_per_symbol

    def modulate(self, bits: Bits) -> Signal:
        """Return the line-coded samples for ``bits``."""
        # NOTE: bits.shape is (n,).
        raise NotImplementedError

    def demodulate(self, signal: Signal) -> Bits:
        """Recover the bit sequence from ``signal``."""
        # NOTE: signal.shape is (n * sps,).
        raise NotImplementedError
```
Iniciamos com uma amplitude e uma quantidade de samples por símbolo.


```python
class NRZPolar(BasebandModulator):
    """Non-Return-to-Zero Polar: 1 -> +V, 0 -> -V, held for the whole bit."""

    def modulate(self, bits: Bits) -> Signal:
        """Map each bit to a constant level held for the whole symbol.

        Args:
            bits: The bit sequence to encode.

        Returns:
            ``+V`` samples for 1s and ``-V`` samples for 0s.
        """
        # +V where the bit is True, -V where it is False.
        levels = np.where(bits, self.amplitude, -self.amplitude)  # (n,)
        # Hold each level for the whole symbol duration.
        # NOTE: np.repeat repeats each element straight after itself.
        # Explicite notation to satisfy mypy.
        signal: Signal = np.repeat(
            levels, self.samples_per_symbol
        )  # (n * sps,)
        return signal
```
Modular é simples: já que o sinal de bits é um booleano, podemos utilizar a função `np.where` e redimensionar os valores. `np.repeat` repete os valores sequencialmente, por exemplo `[1, 2, 3]` -> `[1, 1, 2, 2, 3, 3]`.

```python
def demodulate(self, signal: Signal) -> Bits:
    """Recover bits by thresholding one sample per symbol at 0 V.

    Args:
        signal: The received samples.

    Returns:
        A 1 wherever the level is positive, a 0 otherwise.
    """
    # Level is constant across a symbol; one sample per bit suffices.
    # NOTE: The `::` notation takes every N-th element,
    # starting with the first (index 0).
    symbols = signal[:: self.samples_per_symbol]  # (n,)
    bits: Bits = [bool(sample > 0) for sample in symbols]  # (n,)
    return bits
```
Demodular é mais fácil: pulamos de sps em sps e aceitamos como verdadeiro apenas o que está acima do eixo positivo.


```python
class Manchester(BasebandModulator):
    """Manchester: each bit is a mid-bit transition (data XOR clock).

    Polar convention (matches :class:`NRZPolar`): the signal swings between +V
    and -V, so its average voltage is zero. The clock is low for the first half
    of every bit and high for the second half; XOR-ing the held bit with that
    clock yields a rising edge for a 1 and a falling edge for a 0.
    """

    def _clock(self, symbol_count: int) -> npt.NDArray[np.bool_]:
        """Build the half-bit clock: low first half, high second half.

        Args:
            symbol_count: Number of bits/symbols the clock must cover.

        Returns:
            A boolean signal ``symbol_count * samples_per_symbol`` long.
        """
        half = self.samples_per_symbol // 2
        # One symbol of clock, e.g. [F, F, T, T] for samples_per_symbol == 4.
        one_symbol = [False] * half + [True] * (
            self.samples_per_symbol - half
        )  # (sps,)
        # NOTE: np.tile repeats the whole array, different than np.repeat.
        clock: npt.NDArray[np.bool_] = np.tile(
            np.array(one_symbol), symbol_count
        )  # (symbol_count * sps,)
        return clock
```
Uma função de clock em comum: cria um array de subida e descida para o clock e o repete sequencialmente.

```python
    def modulate(self, bits: Bits) -> Signal:
        """Encode each bit as a mid-bit transition (XOR with the clock).

        Args:
            bits: The bit sequence to encode.

        Returns:
            A polar (+V/-V) signal with one transition per bit.
        """
        # Hold each bit for the whole symbol, then XOR with the half-bit clock.
        bits_held = np.repeat(bits, self.samples_per_symbol)  # (n * sps,)
        high = np.bitwise_xor(bits_held, self._clock(len(bits)))  # (n * sps,)
        # High half -> +V, low half -> -V.
        signal: Signal = np.where(
            high, self.amplitude, -self.amplitude
        )  # (n * sps,)
        return signal
```
Modular aqui segue a mesma lógica de NRZPolar, contudo desta vez fazemos um bitwise XOR com o clock.

```python
def demodulate(self, signal: Signal) -> Bits:
    """Undo the clock XOR and read back one sample per symbol.

    Args:
        signal: The received samples.

    Returns:
        The recovered bit sequence.
    """
    symbol_count = len(signal) // self.samples_per_symbol
    # Which samples sit in the high (+V) half of their bit.
    high = signal > 0  # (n * sps,)
    # NOTE: XOR is its own inverse, so this rebuilds the held bits.
    bits_held = np.bitwise_xor(high, self._clock(symbol_count))  # (n * sps,)
    # Collapse each symbol back to a single bit.
    symbols = bits_held[:: self.samples_per_symbol]  # (n,)
    bits: Bits = [bool(bit) for bit in symbols]  # (n,)
    return bits
```
Demodular funciona de forma idêntica, contudo passamos o bitwise XOR de novo, pois é sua função inversa.

```python
class Bipolar(BasebandModulator):
    """Bipolar (AMI): 0 -> 0 V, 1 -> marks alternating between +V and -V.

    Only the 1s ("marks") carry a pulse, and consecutive marks flip sign: the
    first 1 is +V, the next 1 is -V, and so on, while 0s stay at 0 V and never
    affect the alternation. This keeps the average voltage near zero and lets a
    receiver flag "bipolar violations" (two same-sign marks in a row).
    """

    def modulate(self, bits: Bits) -> Signal:
        """Encode 0 as 0 V and each 1 as a sign-alternating pulse.

        Args:
            bits: The bit sequence to encode.

        Returns:
            A three-level (+V/0/-V) signal.
        """
        bit_array = np.array(bits)  # (n,)
        # NOTE: np.cumsum (cumulative sum) also works for bolean arrays,
        # treating True as 1 and False as 0.
        mark_index = np.cumsum(bit_array)  # (n,)
        # Odd  -> +1; Even -> -1.
        sign = np.where(mark_index % 2 == 1, 1.0, -1.0)  # (n,)
        # 0 bits stay at 0 V; 1 bits take the alternating signed amplitude.
        levels = np.where(bit_array, sign * self.amplitude, 0.0)  # (n,)
        # Hold each level for the whole symbol duration.
        signal: Signal = np.repeat(
            levels, self.samples_per_symbol
        )  # (n * sps,)
        return signal
```

Aqui usamos soma cumulativa. Felizmente o NumPy é muito bem feito e aceita soma de booleanos da mesma maneira que binários. Atenção que ainda passamos `bit_array` para pegar os levels, pois `sign` não é mais booleano e também perdeu o conhecimento dos locais onde não há sinal.

```python
def demodulate(self, signal: Signal) -> Bits:
    """Recover bits from pulse magnitude; the sign carries no data.

    Args:
        signal: The received samples.

    Returns:
        A 1 wherever a pulse is present (either polarity), a 0 otherwise.
    """
    # Level is constant across a symbol; one sample per bit suffices.
    symbols = signal[:: self.samples_per_symbol]  # (n,)
    # A mark sits near +/-V; a 0 near 0 V. Threshold the magnitude halfway.
    bits: Bits = [
        bool(abs(sample) > self.amplitude / 2) for sample in symbols
    ]  # (n,)
    return bits
```
Demodular é simples, basta pegar o valor absoluto e pronto.

### carrier

```python
class CarrierModulator(Modulator):
    """Shared carrier parameters for the concrete modulators below."""

    def __init__(
        self,
        amplitude: float,
        samples_per_symbol: int,
        carrier_frequency: float,
        sample_rate: float,
    ) -> None:
        """Store the carrier parameters.

        Args:
            amplitude: Carrier amplitude in volts.
            samples_per_symbol: Discrete samples emitted per symbol.
            carrier_frequency: Carrier frequency in hertz.
            sample_rate: Sampling rate in hertz.
        """
        self.amplitude = amplitude
        self.samples_per_symbol = samples_per_symbol
        self.carrier_frequency = carrier_frequency
        self.sample_rate = sample_rate

    def _angle(self, frequency: float, symbol_count: int) -> Signal:
        """Carrier angle 2*pi*f*t, with the phase reset every symbol.

        Args:
            frequency: Tone frequency in hertz.
            symbol_count: Number of symbols the angle must cover.

        Returns:
            Angle samples, ``symbol_count * samples_per_symbol`` long.
        """
        # Local time within one symbol, repeated for every symbol.
        symbol_time = (
            np.arange(self.samples_per_symbol) / self.sample_rate
        )  # (sps,)
        time = np.tile(symbol_time, symbol_count)  # (symbol_count * sps,)
        angle: Signal = 2 * np.pi * frequency * time  # (symbol_count * sps,)
        return angle  # (symbol_count * sps,)

    def _symbols(self, signal: Signal) -> Signal:
        """Reshape a flat signal into one row of samples per symbol."""
        blocks: Signal = signal.reshape(
            -1, self.samples_per_symbol
        )  # (signal.shape[0] // sps, sps,)
        return blocks

    def modulate(self, bits: Bits) -> Signal:
        """Modulate ``bits`` onto the carrier."""
        # NOTE: bits.shape is (n,).
        raise NotImplementedError

    def demodulate(self, signal: Signal) -> Bits:
        """Recover bits from the modulated carrier ``signal``."""
        # NOTE: signal.shape is (n * sps,).
        raise NotImplementedError
```
Função compartilhada para gerar ângulos e função auxiliar para ajudar na hora de inverter sinais.

```python
class ASK(CarrierModulator):
    """Amplitude Shift Keying: bit value selects the carrier amplitude."""

    def modulate(self, bits: Bits) -> Signal:
        """Scale the carrier amplitude by each bit (full for 1, zero for 0).

        Args:
            bits: The bit sequence to encode.

        Returns:
            The on/off-keyed carrier: ``A sin(2*pi*f*t)`` during a 1, 0 during
            a 0.
        """
        carrier = self.amplitude * np.sin(
            self._angle(self.carrier_frequency, len(bits))
        )  # (n * sps,)
        keep = np.repeat(bits, self.samples_per_symbol)  # (n * sps,)
        # Keep the carrier during a 1, blank it during a 0 (on/off keying).
        signal: Signal = np.where(keep, carrier, 0.0)  # (n * sps,)
        return signal
```
Nada de especial aqui, `np.where` faz a mágica. O fato de já estarmos usando booleanos ajuda muito.

```python
def demodulate(self, signal: Signal) -> Bits:
    """Match each symbol against the carrier: strong response means 1.

    Args:
        signal: The received samples.

    Returns:
        The recovered bit sequence.
    """
    reference = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
    # Correlate each symbol with the reference.
    correlation = self._symbols(signal) @ reference  # (n, sps) @ (sps,) -> (n,)
    # A "1" correlates to ~ A * sps / 2 (average energy of a sine wave),
    # a "0" to ~0; threshold halfway.
    # NOTE: A*sum{n=0}^{sps-1}sin^2(2*pi*n*f/sr)~ A*sps/2,
    # if f*sps/sr is an integer.
    # threshold = self.amplitude * self.samples_per_symbol / 4
    # True threshold, but slower to compute.
    threshold = self.amplitude * reference @ reference / 2  # scalar
    bits: Bits = [bool(value > threshold) for value in correlation]  # (n,)
    return bits
```
Ao invés de ficar refém de valores inteiros na divisão de frequência e símbolos, podemos fazer na força bruta e aceitar metade do valor de referência.

```python
class FSK(CarrierModulator):
    """Frequency Shift Keying: bit value selects the carrier frequency.

    A 0 is sent at ``carrier_frequency`` and a 1 at twice that frequency.
    """

    def modulate(self, bits: Bits) -> Signal:
        """Send each bit as a tone: low frequency for 0, double for 1.

        Args:
            bits: The bit sequence to encode.

        Returns:
            The frequency-keyed carrier.
        """
        # Default waveform.
        tone0 = np.sin(
            self._angle(self.carrier_frequency, len(bits))
        )  # (n * sps,)
        # Double frequency waveform.
        tone1 = np.sin(
            self._angle(2 * self.carrier_frequency, len(bits))
        )  # (n * sps,)
        held = np.repeat(bits, self.samples_per_symbol)  # (n * sps,)
        signal: Signal = self.amplitude * np.where(
            held, tone1, tone0
        )  # (n * sps,)
        return signal
```
Novamente o `np.where` carregando.

```python
def demodulate(self, signal: Signal) -> Bits:
    """Correlate each symbol with both tones; the stronger one wins.

    Args:
        signal: The received samples.

    Returns:
        The recovered bit sequence.
    """
    blocks = self._symbols(signal)  # (n, sps,)
    # Default waveform.
    ref0 = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
    # Double frequency waveform.
    ref1 = np.sin(self._angle(2 * self.carrier_frequency, 1))  # (sps,)
    # Take np.abs, we only want to compare the strength of the correlation,
    # not its sign.
    corr0 = np.abs(blocks @ ref0)  # (n, sps) @ (sps,) -> (n,)
    corr1 = np.abs(blocks @ ref1)  # (n, sps) @ (sps,) -> (n,)
    bits: Bits = [bool(value) for value in corr1 > corr0]  # (n,)
    return bits
```
O valor que mais se correlaciona no produto interno ganha, simples assim.

```python
class QPSK(CarrierModulator):
    """Quadrature PSK: each symbol carries two bits as a carrier phase.

    The first bit picks the in-phase (cosine) sign, the second the quadrature
    (sine) sign; a False bit maps to +1 and a True bit to -1.
    """

    def modulate(self, bits: Bits) -> Signal:
        """Encode bit pairs as the sign of the cosine and sine components.

        Args:
            bits: The bit sequence to encode (length a multiple of 2).

        Returns:
            The phase-keyed carrier.
        """
        # I * cos(w) - Q * sin(w) = sqrt(I^2+Q^2) * sin(w + atan2(I, Q)).
        # QPSK(1), I,Q in [-1, 1] -> atan2(I, Q) = (45, 135, 225, 315) degrees.
        # NOTE: (n,) is even, so we dont get errors when reshaping.
        pairs = np.array(bits).reshape(-1, 2)  # (n // 2, 2)
        in_phase = np.where(pairs[:, 0], -1.0, 1.0)  # (n // 2,)
        quadrature = np.where(pairs[:, 1], -1.0, 1.0)  # (n // 2,)
        cosine = np.cos(self._angle(self.carrier_frequency, 1))  # (sps,)
        sine = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
        # Divide by sqrt(2) to keep the total power A^2 constant across
        # all four constellation points.
        scale = self.amplitude / np.sqrt(2)
        # Combine I and Q into one symbol each, then flatten to a signal.
        symbols = (
            in_phase[:, np.newaxis] * cosine - quadrature[:, np.newaxis] * sine
        )  # (n // 2, sps)
        signal: Signal = scale * symbols.reshape(-1)  # (n // 2 * sps,)
        return signal
```
Dividimos em duplas `[[a, b], [a, b]]`. O `reshape` é mágico e consegue dividir o sinal desta maneira, `shape(y, x)`. `np.newaxis` é `None` e basicamente acopla todo o valor do cosseno ou seno naquela área.

```python
def demodulate(self, signal: Signal) -> Bits:
    """Recover each bit pair by jointly de-mixing the I and Q correlations.

    The matched filter assumes the cosine and sine references are orthogonal
    over one symbol, which only holds for a whole number of carrier cycles
    per symbol. To stay correct for any carrier, this solves the 2x2 system
    coupling the in-phase (I) and quadrature (Q) components, instead of
    trusting the raw correlation signs.

    Args:
        signal: The received samples.

    Returns:
        The recovered bit sequence.
    """
    blocks = self._symbols(signal)  # (n // 2, sps)
    cosine = np.cos(self._angle(self.carrier_frequency, 1))  # (sps,)
    sine = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
    # Idea: the received symbol is a linear combination
    # of the two references:
    # [cc, cs] = [I, Q] @ [[a, -b], [b, -d]]
    # -> [I, Q] = [cc, cs] @ [[-d, b], [-b, a]] / det
    a = cosine @ cosine  # scalar
    b = sine @ cosine  # scalar
    d = sine @ sine  # scalar
    # Each symbol is scale * (I * cosine - Q * sine),
    cc = blocks @ cosine  # (n // 2,)
    cs = blocks @ sine  # (n // 2,)
    det = b * b - a * d  # det([[a, -b], [b, -d]]); < 0 for independent refs
    in_phase = (-d * cc + b * cs) / det  # (n // 2,)
    quadrature = (-b * cc + a * cs) / det  # (n // 2,)
    # I, Q < 0 encode a set bit (1), I, Q >= 0 encode a clear bit (0).
    first = in_phase < 0
    second = quadrature < 0
    pairs = np.stack([first, second], axis=1)  # (n // 2, 2)
    bits: Bits = [bool(value) for value in pairs.reshape(-1)]  # (n,)
    return bits
```
A demodulação aqui é interessante, fazemos a matriz inversa para conseguir obter os valores.

$$
\begin{bmatrix}
    cc\\
    cs
\end{bmatrix} = \begin{bmatrix}
    \cos^2(\omega) & -\cos(\omega)\cdot\sin(\omega) \\
    \sin(\omega)\cdot\cos(\omega) & -\sin^2(\omega)
\end{bmatrix}\ \begin{bmatrix}
    I \\
    Q
\end{bmatrix}
$$
Mesma ideia de força bruta utilizada no ASK, contudo desta vez um pouco mais sofisticada. Nós sabemos com certeza absoluta que o determinante não será zero, Cauchy-Schwarz.

```python
class QAM16(CarrierModulator):
    """16-QAM (CF-13): each symbol carries four bits as one constellation point.

    The quadbit is laid out as
    ``[I sign, Q sign, I magnitude, Q magnitude]`` (sign bits first, then
    magnitudes). Each axis rides the Gray-coded ladder ``{-3, -1, +1, +3}``:
    the sign bit picks +/-, the magnitude bit picks the inner (1) or outer (3)
    ring. The constellation is normalised so the outer corner reaches the
    configured amplitude ``A``, which yields the three envelope levels
    ``A/3``, ``A*sqrt(5)/3`` and ``A`` (~0.33, 0.75, 1.00 for ``A = 1``).
    """

    def modulate(self, bits: Bits) -> Signal:
        """Map each group of four bits to one amplitude/phase point.

        Args:
            bits: The bit sequence to encode (length a multiple of 4).

        Returns:
            The quadrature-amplitude-modulated carrier.
        """
        # [I sign, Q sign, I magnitude, Q magnitude].
        groups = np.array(bits).reshape(-1, 4)  # (n // 4, 4)
        in_level = self._levels(groups[:, 0], groups[:, 2])  # (n // 4,)
        quad_level = self._levels(groups[:, 1], groups[:, 3])  # (n // 4,)
        cosine = np.cos(self._angle(self.carrier_frequency, 1))  # (sps,)
        sine = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
        # Normalise so the outer corner (3, 3) reaches amplitude A:
        # unit * sqrt(3**2 + 3**2) = unit * 3*sqrt(2) = A.
        unit = self.amplitude / (3 * np.sqrt(2))
        # symbol = unit*(I*cos - Q*sin); phase = atan2(Q, I).
        symbols = (
            in_level[:, np.newaxis] * cosine - quad_level[:, np.newaxis] * sine
        )  # (n // 4, sps)
        signal: Signal = unit * symbols.reshape(-1)  # (n // 4 * sps,)
        return signal
```
```python
def demodulate(self, signal: Signal) -> Bits:
    """Recover the I/Q levels by de-mixing the correlations, then decode.

    Like the QPSK receiver, this solves the 2x2 system coupling the cosine
    and sine references, so it stays correct even when the carrier does not
    complete a whole number of cycles per symbol. It then thresholds each
    recovered level on the {-3, -1, +1, +3} ladder.

    Args:
        signal: The received samples.

    Returns:
        The recovered bit sequence.
    """
    blocks = self._symbols(signal)  # (n // 4, sps)
    cosine = np.cos(self._angle(self.carrier_frequency, 1))  # (sps,)
    sine = np.sin(self._angle(self.carrier_frequency, 1))  # (sps,)
    # Idea: the received symbol is a linear combination
    # of the two references:
    # [cc, cs] = unit * [I, Q] @ [[a, -b], [b, -d]]
    # -> [I, Q] = [cc, cs] @ [[-d, b], [-b, a]] / (det * unit)
    a = cosine @ cosine  # scalar
    b = sine @ cosine  # scalar
    d = sine @ sine  # scalar
    cc = blocks @ cosine  # (n // 4,)
    cs = blocks @ sine  # (n // 4,)
    det = b * b - a * d  # det([[a, -b], [b, -d]]); < 0 for independent refs
    unit = self.amplitude / (3 * np.sqrt(2))
    in_level = (-d * cc + b * cs) / (det * unit)  # (n // 4,)
    quad_level = (-b * cc + a * cs) / (det * unit)  # (n // 4,)
    # _bits returns [sign, magnitude]; quadbit order
    # [I sign, Q sign, I magnitude, Q magnitude].
    i_bits = self._bits(in_level)  # (n // 4, 2)
    q_bits = self._bits(quad_level)  # (n // 4, 2)
    quadbits = np.stack(
        [i_bits[:, 0], q_bits[:, 0], i_bits[:, 1], q_bits[:, 1]], axis=1
    )  # (n // 4, 4)
    bits: Bits = [bool(value) for value in quadbits.reshape(-1)]  # (n,)
    return bits


@staticmethod
def _levels(sign_bit: Signal, magnitude_bit: Signal) -> Signal:
    """Map a sign bit and a magnitude bit to a level in {-3,-1,+1,+3}.

    Per CF-13: sign bit ``1 -> +``, ``0 -> -``; magnitude bit
    ``1 -> outer (3)``, ``0 -> inner (1)``. Gray-coded along the ladder.
    """
    sign = np.where(sign_bit, 1.0, -1.0)
    magnitude = np.where(magnitude_bit, 3.0, 1.0)
    levels: Signal = sign * magnitude
    return levels


@staticmethod
def _bits(level: Signal) -> npt.NDArray[np.bool_]:
    """Invert :meth:`_levels`: recover the ``[sign, magnitude]`` bits."""
    sign_bit = level > 0  # positive level -> 1
    magnitude_bit = np.abs(level) > 2  # outer (3) -> 1, inner (1) -> 0
    pairs: npt.NDArray[np.bool_] = np.stack([sign_bit, magnitude_bit], axis=1)
    return pairs
```

## link

### framing

```python
# ./src/layer_manager/link/framing.py
"""Framing schemes for the data-link layer.

Each class satisfies :class:`layer_manager.protocol.Framer`: it concatenates
payloads into a delimited bit stream and recovers them on the other side.
See the "Enquadramento" reference material for the exact field layouts.
"""

from itertools import chain

import numpy as np
import numpy.typing as npt

from layer_manager.protocol import Framer
from layer_manager.types import Bits


class FramerBase(Framer):
    """Shared base for the concrete framers.

    Mirrors the physical layer's ``CarrierModulator``/``BasebandModulator``:
    it declares the :class:`~layer_manager.protocol.Framer` interface for
    subclasses to override and holds the bit/integer conversion helpers they
    reuse.
    """

    def frame(self, payloads: list[Bits]) -> Bits:
        """Delimit and concatenate ``payloads`` into one bit stream."""
        raise NotImplementedError

    def deframe(self, stream: Bits) -> list[Bits]:
        """Recover the original payloads from a framed ``stream``."""
        raise NotImplementedError

    @staticmethod
    def _encode(count: list[int]) -> npt.NDArray[np.bool_]:
        """Return ``count`` as a fixed-width bool array, MSB first."""
        return np.unpackbits(np.array(count, dtype=np.uint8)).astype(dtype=bool)

    @staticmethod
    def _decode(bits: Bits) -> npt.NDArray[np.uint8]:
        """Return the integer represented by the fixed-width bool array."""
        return np.packbits(np.array(bits, dtype=np.uint8))


class CharCountFramer(FramerBase):
    """Prefixes each frame with a header counting its bytes.

    Each frame is ``[count header][payload]``: a fixed-width header states how
    many bytes the payload holds, so the receiver reads a count, takes that
    many bytes, and repeats. Payloads are byte-aligned ``Bits`` (length a
    multiple of 8, MSB first), so the count is in bytes.

    Design choice: the count covers the payload bytes only (the header is
    separate) and is ``HEADER_BITS`` wide. Known weakness to note: a single
    corrupted count desynchronises every following frame.
    """

    HEADER_BITS = 8  # one byte -> counts up to 255 payload bytes per frame

    def frame(self, payloads: list[Bits]) -> Bits:
        """Prepend a length header to each payload and concatenate.

        Steps:
          1. For each ``payload``, compute its length in bytes
             (``len(payload) // 8``).
          2. Encode that count as ``HEADER_BITS`` bools, MSB first.
          3. Concatenate ``header + payload`` for every frame into one stream.
        """
        new_payload: list[Bits] = [
            self._encode([len(frame) // 8]).tolist() + frame
            for frame in payloads
        ]
        bits: Bits = [bits for sublist in new_payload for bits in sublist]
        return bits

    def deframe(self, stream: Bits) -> list[Bits]:
        """Read each length header to slice the stream back into payloads.

        Inverse of :meth:`frame`. Steps:
          1. Walk the stream with a cursor starting at 0.
          2. Decode ``HEADER_BITS`` bits (MSB first) into ``count`` bytes.
          3. Take the next ``count * 8`` bits as the payload.
          4. Advance past header + payload; repeat until the stream is consumed.
        """
        cursor = 0

        def take(n: int) -> Bits:
            nonlocal cursor
            chunk = stream[cursor : cursor + n]
            cursor += n
            return chunk

        payloads: list[Bits] = []
        while cursor < len(stream):
            count = int(self._decode(take(self.HEADER_BITS))[0])
            payloads.append(take(count * 8))
        return payloads


class ByteStuffingFramer(FramerBase):
    """Delimits frames with FLAG bytes, escaping FLAG/ESC bytes in the payload.

    Each frame is ``FLAG | stuffed(payload) | FLAG``. Inside the payload any
    byte equal to FLAG or ESC is prefixed with an ESC byte, so a bare FLAG in
    the stream is always a real delimiter. Payloads are byte-aligned, so
    stuffing operates on whole bytes (convert with np.packbits/np.unpackbits).

    Design choice: simple insertion (the escaped byte is left unchanged, not
    XOR'd like PPP); FLAG/ESC are the HDLC-style 0x7E / 0x7D.
    """

    FLAG = 0x7E  # frame delimiter byte (0b0111_1110)
    ESC = 0x7D  # escape byte (0b0111_1101)

    def frame(self, payloads: list[Bits]) -> Bits:
        """Escape payload FLAG/ESC bytes and wrap each frame in FLAG bytes.

        Each frame is: opening FLAG, the payload bytes (with FLAG/ESC bytes
        escaped by a preceding ESC), then a closing FLAG.
        """
        stuffed: list[int] = []
        for payload in payloads:
            data = self._decode(payload)
            needs_esc = (data == self.FLAG) | (
                data == self.ESC
            )  # data.shape = (n,)

            pairs = np.empty((len(data), 2), dtype=data.dtype)  # (n, 2)
            pairs[:, 0] = self.ESC
            pairs[:, 1] = data
            # Ravel is faster than flatten and we don't need
            # a copy since we'll slice it right away;
            flat = pairs.ravel()  # (2n,)

            keep = np.ones(len(flat), dtype=bool)  # (2n,)
            keep[0::2] = (
                needs_esc  # keep the ESC only if the byte needs escaping
            )
            body = flat[keep]  # (n + number of bytes that need escaping,)

            stuffed.append(self.FLAG)
            stuffed.extend(body.tolist())
            stuffed.append(self.FLAG)

        bits: Bits = self._encode(stuffed).tolist()
        return bits

    def deframe(self, stream: Bits) -> list[Bits]:
        """Split on FLAG bytes and undo the byte stuffing.

        Inverse of :meth:`frame`. A bare FLAG opens or closes a frame; inside a
        frame an ESC marks the next byte as literal data (the ESC is dropped).
        """
        payloads: list[Bits] = []
        current: list[int] = []
        in_frame = False
        escape_next = False

        for byte in self._decode(stream).tolist():
            if not in_frame:
                if byte == self.FLAG:  # opening FLAG
                    in_frame = True
                continue

            if escape_next:
                current.append(byte)
                escape_next = False
            elif byte == self.ESC:
                escape_next = True
            elif byte == self.FLAG:  # closing FLAG
                payloads.append(self._encode(current).tolist())
                current = []
                in_frame = False
            else:
                current.append(byte)

        return payloads


class BitStuffingFramer(FramerBase):
    """Delimits frames with a FLAG bit pattern, stuffing bits to avoid it."""

    FLAG = 0x7E  # frame delimiter byte (0b0111_1110)

    def frame(self, payloads: list[Bits]) -> Bits:
        """Stuff a 0 after every run of five 1s; wrap each frame in FLAGs."""
        flag: Bits = self._encode([self.FLAG]).tolist()
        stream: Bits = []
        for payload in payloads:
            stream += flag  # opening FLAG
            ones = 0
            for bit in payload:
                stream.append(bit)
                if bit:
                    ones += 1
                    if ones == 5:
                        stream.append(False)  # insert stuffing 0
                        ones = 0
                else:
                    ones = 0
            stream += flag  # closing FLAG
        return stream

    def deframe(self, stream: Bits) -> list[Bits]:
        """Split on FLAG patterns and remove stuffed bits (inverse of frame)."""
        bits = np.array(stream, dtype=bool)
        flag_bits = np.asarray(self._encode([self.FLAG]), dtype=bool)
        flag_len = len(flag_bits)

        if bits.size < flag_len:  # empty / too-short stream: no frames
            return []

        # Locate FLAG patterns via a sliding-window equality check.
        windows = np.lib.stride_tricks.sliding_window_view(
            bits, flag_len
        )  #  (N - flag_len + 1, flag_len)
        matches = (windows == flag_bits).all(axis=1)  # (n_windows,) bool
        flag_indices = np.where(matches)[0]  # (n_flags,) indices of FLAG starts

        if len(flag_indices) % 2 != 0:
            raise ValueError(
                "Invalid frame structure: unbalanced FLAG patterns"
            )

        payloads: list[Bits] = []
        for i in range(0, len(flag_indices), 2):
            start = flag_indices[i] + flag_len
            end = flag_indices[i + 1]
            payloads.append(self._unstuff(bits[start:end]))
        return payloads

    @staticmethod
    def _unstuff(payload: npt.NDArray[np.bool_]) -> Bits:
        """Drop the 0 that follows each run of five consecutive 1s."""
        out: list[bool] = []
        ones = 0
        it = iter(payload.tolist())
        for bit in it:
            out.append(bit)
            if bit:
                ones += 1
                if ones == 5:
                    next(it, None)  # skip the stuffed 0
                    ones = 0
            else:
                ones = 0
        return out


def chunk(bits: Bits, max_frame_size: int) -> list[Bits]:
    """Split a flat bit stream into frames of at most max_frame_size bytes.

    Args:
        bits: The application bit stream (length a multiple of 8).
        max_frame_size: Maximum payload per frame, in bytes.

    Returns:
        The per-frame payloads, in order; the last may be shorter. Empty
        input gives an empty list (no frames).
    """
    step = max_frame_size * 8  # bytes -> bits per full frame
    return [bits[i : i + step] for i in range(0, len(bits), step)]


def join(frames: list[Bits]) -> Bits:
    """Concatenate recovered frame payloads back into one bit stream.

    Inverse of :func:`chunk` (after framing/EDC have been stripped).
    """
    return list(chain.from_iterable(frames))
```

### detection

```python
# ./src/layer_manager/link/detection.py
"""Error-detection schemes for the data-link layer.

Each class satisfies :class:`layer_manager.protocol.ErrorDetector`. None of
these may rely on an external library for the calculation itself (e.g. ``zlib``
for CRC); the algorithm must be implemented here.
"""

from layer_manager.protocol import ErrorDetector
from layer_manager.types import Bits


class DetectorBase(ErrorDetector):
    """Shared base for the concrete error detectors.

    Mirrors framing's ``FramerBase``: it declares the
    :class:`~layer_manager.protocol.ErrorDetector` interface for subclasses to
    override and holds the bit/integer conversion helpers shared by the
    checksum and CRC schemes.
    """

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with its error-detecting code appended."""
        raise NotImplementedError

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Validate a received block and strip its error-detecting code."""
        raise NotImplementedError

    @staticmethod
    def _int_to_bits(value: int, width: int) -> Bits:
        """Write ``value`` as ``width`` bits, MSB first."""
        return [bool((value >> (width - 1 - i)) & 1) for i in range(width)]

    @staticmethod
    def _bits_to_int(bits: Bits) -> int:
        """Read a bit list (MSB first) as an unsigned integer."""
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return value


class ParityDetector(DetectorBase):
    """Appends one even-parity bit per block.

    Even parity: the appended bit is chosen so the **total** number of ``1``
    bits in the block (payload + parity bit) is even. On receipt, a block whose
    ``1`` count is odd must have been corrupted.

    This is the cheapest detector and also the weakest: it only catches an
    **odd** number of bit flips. Two flips (or any even number) leave the
    parity even and slip through undetected. Note it appends a single bit, so
    the result is no longer byte-aligned.
    """

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with an even-parity bit appended.

        Args:
            data: The payload bits to protect.

        Returns:
            ``data`` followed by one parity bit: ``1`` when ``data`` has an odd
            number of ones (to even it out), ``0`` otherwise.
        """
        parity = sum(data) % 2
        bits: Bits = [*data, bool(parity)]
        return bits

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Verify the parity bit and strip it from ``data``.

        Args:
            data: A received block, parity bit included (as produced by
                :meth:`encode`).

        Returns:
            ``(payload, ok)`` where ``payload`` is ``data`` without its trailing
            parity bit and ``ok`` is ``True`` when the block's total ``1`` count
            is even (no odd-sized error detected).
        """
        ok = sum(data) % 2 == 0
        return data[:-1], ok


class ChecksumDetector(DetectorBase):
    """Appends the one's-complement checksum of fixed-size blocks.

    The Internet-checksum scheme shown in class: split the data into
    ``block_bits``-wide blocks, add them with **end-around carry** (one's-
    complement addition), and append the one's complement of that sum. On
    receipt, summing every block *including* the checksum gives all ones when
    the data is intact. Catches more than parity (it sees the magnitude of a
    change), but reordered blocks or offsetting errors can still cancel out.
    """

    def __init__(self, block_bits: int) -> None:
        """Store the block size used to split the data.

        Args:
            block_bits: Width, in bits, of each block summed for the checksum.
        """
        self.block_bits = block_bits

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with its block checksum appended.

        Args:
            data: The payload bits to protect.

        Returns:
            ``data`` followed by the ``block_bits``-wide one's-complement
            checksum of its blocks.
        """
        total = self._ones_complement_sum(self._blocks(data))
        mask = (1 << self.block_bits) - 1  # 2^block_bits - 1
        checksum = ~total & mask  # one's complement, kept to block_bits
        return [*data, *self._int_to_bits(checksum, self.block_bits)]

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Recompute the checksum to validate and strip it from ``data``.

        Args:
            data: A received block, checksum included (as produced by
                :meth:`encode`).

        Returns:
            ``(payload, ok)`` where ``payload`` is ``data`` without its trailing
            ``block_bits`` checksum and ``ok`` is ``True`` when the blocks plus
            the checksum sum to all ones (no error detected).
        """
        payload = data[: -self.block_bits]
        received = self._bits_to_int(data[-self.block_bits :])
        total = self._ones_complement_sum([*self._blocks(payload), received])
        ok = total == (1 << self.block_bits) - 1  # data + complement == 1...1
        return payload, ok

    def _blocks(self, bits: Bits) -> list[int]:
        """Split ``bits`` into ``block_bits``-wide ints (last zero-padded)."""
        pad = (-len(bits)) % self.block_bits
        padded = [*bits, *([False] * pad)]  # zero padding for the last block
        width = self.block_bits
        return [
            self._bits_to_int(padded[i : i + width])
            for i in range(0, len(padded), width)
        ]

    def _ones_complement_sum(self, values: list[int]) -> int:
        """Add ``values`` with end-around carry, folded to ``block_bits``."""
        mask = (1 << self.block_bits) - 1
        total = sum(values)
        # ones complement addition:
        # fold every overflow bit back into the low bits
        while total > mask:
            total = (total & mask) + (total >> self.block_bits)
        return total


class CRC32Detector(DetectorBase):
    """Appends a CRC-32 (IEEE 802 / ISO-HDLC) remainder.

    Treats the bit stream as a polynomial over GF(2) and appends the remainder
    of dividing it by the CRC-32 generator; a received block is intact when its
    recomputed remainder matches. This is the strongest detector here -- it
    catches every burst error up to 32 bits, every odd number of bit flips, and
    the overwhelming majority of longer bursts.

    Uses the canonical CRC-32/ISO-HDLC parameters (as in Ethernet and zip):
    reflected generator ``0xEDB88320``, register preset to ``0xFFFFFFFF``, and a
    final XOR with ``0xFFFFFFFF``. With those, the standard check value of
    ``"123456789"`` is ``0xCBF43926``. The division is hand-rolled (no ``zlib``)
    and processed byte-by-byte to match the reflected convention.
    """

    # <https://media.neliti.com/media/publications/501671-analysis-and-design-of-crc-32-ieee-8023-d727547b.pdf>
    # _POLY = 11101101101110001000001100100000,
    # LSB first (reflected form of 0x04C11DB7)
    _POLY = 0xEDB88320  # reflected CRC-32 generator (IEEE 802 / ISO-HDLC)
    _INIT = 0xFFFFFFFF  # register preset (all ones)
    _XOROUT = 0xFFFFFFFF  # final inversion applied to the register
    _WIDTH = 32  # remainder width, in bits

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with its 32-bit CRC remainder appended.

        Args:
            data: The payload bits to protect.

        Returns:
            ``data`` followed by the 32-bit CRC remainder, MSB first.
        """
        crc = self._crc(data)
        return list(data) + self._int_to_bits(crc, self._WIDTH)

    def check(self, data: Bits) -> tuple[Bits, bool]:
        """Recompute the CRC to validate and strip it from ``data``.

        Args:
            data: A received block, CRC included (as produced by
                :meth:`encode`).

        Returns:
            ``(payload, ok)`` where ``payload`` is ``data`` without its trailing
            32-bit CRC and ``ok`` is ``True`` when the recomputed CRC matches
            the received one. A block shorter than the CRC is rejected as
            ``(data, False)``.
        """
        if len(data) < self._WIDTH:
            return list(data), False
        payload = list(data[: -self._WIDTH])
        received = data[-self._WIDTH :]
        expected = self._int_to_bits(self._crc(payload), self._WIDTH)
        return payload, received == expected

    def _crc(self, data: Bits) -> int:
        """Compute the CRC-32 register over ``data`` (MSB-first bits).

        Packs the bits into bytes (MSB first), folds each through
        :meth:`_update_byte`, then applies the final XOR. A trailing partial
        byte is zero-padded on the right.

        Args:
            data: The bits to run the CRC over.

        Returns:
            The 32-bit CRC value.
        """
        crc = self._INIT
        bit_count = 0
        byte = 0
        for bit in data:
            byte = (byte << 1) | bit
            bit_count += 1
            # Upon a full byte, fold it into the CRC register
            # and reset for the next one.
            if bit_count == 8:
                crc = self._update_byte(crc, byte)
                byte = 0
                bit_count = 0
        # Fold a trailing partial byte, if any, padded with zeros on the right.
        if bit_count:
            byte <<= 8 - bit_count  # left-align a trailing partial byte
            crc = self._update_byte(crc, byte)
        # Apply inversion and bound to 32 bits.
        return crc ^ self._XOROUT

    def _update_byte(self, crc: int, byte: int) -> int:
        """Fold one byte into the running CRC register.

        The reflected generator and right-shifts already encode the bit
        reflection, so the byte is fed at its natural value (no reversal).

        Args:
            crc: The current 32-bit register.
            byte: The next message byte (0-255).

        Returns:
            The updated 32-bit register.
        """
        # Folds byte into the crc.
        crc ^= byte
        for _ in range(8):
            # if remainder is 1.
            if crc & 1:
                crc = (crc >> 1) ^ self._POLY  # Subtract the divisor poly.
            else:
                crc >>= 1
        # Mask back to 32 bits in case of Python's unbounded int.
        return crc & 0xFFFFFFFF
```

### correction

```python
# ./src/layer_manager/link/correction.py
"""Error-correction schemes for the data-link layer.

Satisfies :class:`layer_manager.protocol.ErrorCorrector`.
"""

from layer_manager.protocol import ErrorCorrector
from layer_manager.types import Bits


class HammingCorrector(ErrorCorrector):
    """Hamming single-error correction (SEC).

    Interleaves parity bits at the power-of-two positions (1, 2, 4, 8, ...) of
    a 1-indexed codeword; the data fills the remaining positions. Each parity
    bit ``2^i`` covers exactly the positions whose index has bit ``i`` set, so
    on receipt the recomputed parities -- read as a binary number -- spell out
    the **position of the flipped bit** (the *syndrome*); 0 means no error.

    The whole payload is treated as one block: ``r`` parity bits are added with
    ``2^r >= m + r + 1`` for ``m`` data bits, enough to address every position
    plus the no-error case. That corrects **one** bit error per block; a second
    error in the same block is mislocated and "corrected" wrongly (this is SEC,
    not SECDED), so pair it with a detector (e.g. CRC) to catch that case.
    """

    def encode(self, data: Bits) -> Bits:
        """Return ``data`` with Hamming parity bits interleaved.

        Picks the smallest ``r`` with ``2^r >= m + r + 1``, lays the data into
        the non-power-of-two positions of an ``m + r`` bit codeword, then sets
        each power-of-two parity bit to even parity over the positions it
        covers.

        Args:
            data: The payload bits to protect.

        Returns:
            The Hamming codeword: ``data`` with ``r`` parity bits interleaved.
        """
        m = len(data)
        # Find r such that the codeword length m + r fits: 2^r >= m + r + 1.
        r = 0
        while (1 << r) < m + r + 1:
            r += 1
        n = m + r

        # Place data bits into non-power-of-two positions (1-indexed).
        code = [False] * (n + 1)  # index 0 unused
        di = 0
        for pos in range(1, n + 1):
            if not self._is_power_of_two(pos):
                code[pos] = data[di]
                di += 1

        # Each parity bit at 2^i is even parity over the positions (other than
        # itself) whose index has bit i set, which are all data positions.
        for i in range(r):
            p = 1 << i
            parity = False
            for pos in range(1, n + 1):
                # pos & p is True if pos has bit p,
                # pos != p. Excludes itselt.
                if pos & p and pos != p:
                    parity ^= code[pos]
            code[p] = parity

        return code[1:]  # Ignores idx 0.

    def decode(self, data: Bits) -> tuple[Bits, bool]:
        """Locate and flip a corrupted bit, then strip the parity bits.

        Computes the syndrome (XOR of the indices of every set bit): 0 for an
        intact codeword, otherwise the position of the single flipped bit,
        which is corrected in place.

        Args:
            data: A received Hamming codeword (as produced by :meth:`encode`).

        Returns:
            ``(payload, corrected)`` where ``payload`` is the data bits with the
            parity bits removed and ``corrected`` is ``True`` when a single-bit
            error was located and repaired, ``False`` otherwise.
        """
        n = len(data)
        code = [False, *data]  # shift to 1-indexed; index 0 unused

        # Syndrome = XOR of the indices of all set bits. Each bit i of the
        # syndrome is the parity of group 2^i, so an intact codeword gives 0.
        syndrome = 0  # Will point to the flipped bit.
        for pos in range(1, n + 1):
            if code[pos]:
                syndrome ^= pos

        corrected = False
        if 0 < syndrome <= n:  # a locatable single-bit error
            code[syndrome] = not code[syndrome]  # flip it back
            corrected = True

        # Strip parity bits (power-of-two positions) to recover the payload.
        payload = [
            code[pos]
            for pos in range(1, n + 1)
            if not self._is_power_of_two(pos)
        ]
        return payload, corrected

    @staticmethod
    def _is_power_of_two(pos: int) -> bool:
        """Return whether ``pos`` is a power of two (a parity position)."""
        return pos > 0 and (pos & (pos - 1)) == 0
```