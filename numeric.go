package mkparam

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"time"
)

type Threshold[T NumberConstraint] struct {
	value   T
	onCross func(T, T)
}

type Numeric[T NumberConstraint] struct {
	value            T
	format           string         // Format string for custom string representation.
	compareCondition func(T, T) int // Custom comparison function.
	minValue         *T             // Minimum allowable value (if any).
	maxValue         *T             // Maximum allowable value (if any).
	incrementStep    T

	onUpdate    func(T)         // Called when the value is updated.
	onIncrease  func(T, T)      // Called when the value increases.
	onDecrease  func(T, T)      // Called when the value decreases.
	onOperation func(string, T) // Called when an operation is performed.
	onError     func(error)     // Called when an error occurs.
	onOverflow  func(oldValue, newValue T)
	onUnderflow func(oldValue, newValue T)

	thresholds []Threshold[T] // List of thresholds to notify on crossing.
}

func NewNumeric[T NumberConstraint](value T) *Numeric[T] {
	return &Numeric[T]{value: value}
}

func (n *Numeric[T]) Set(value T, errs ...error) *Numeric[T] {
	oldValue := n.value
	// Trigger the onError callback if defined and an error condition is met.
	if n.onError != nil && len(errs) > 0 {
		for _, err := range errs {
			n.onError(err)
		}
	}

	// Apply minimum value constraint if defined.
	if n.minValue != nil && lessThan(value, *n.minValue) {
		if n.onUnderflow != nil {
			n.onUnderflow(value, oldValue)
		}
	}

	// Apply maximum value constraint if defined.
	if n.maxValue != nil && greaterThan(value, *n.maxValue) {
		if n.onOverflow != nil {
			n.onOverflow(value, oldValue)
		}
	}

	// Trigger the onUpdate callback if defined and if the value has changed.
	if n.onUpdate != nil && !equal(value, oldValue) {
		n.onUpdate(value)
	}

	// Trigger the onIncrease callback if defined and if the value has increased.
	if n.onIncrease != nil && greaterThan(value, oldValue) {
		n.onIncrease(value, oldValue)
	}

	// Trigger the onDecrease callback if defined and if the value has decreased.
	if n.onDecrease != nil && lessThan(value, oldValue) {
		n.onDecrease(value, oldValue)
	}

	// Notify about crossing thresholds based on the old and new values.
	n.NotifyThresholds(oldValue, value)

	// Update the value.
	n.value = value

	return n
}

func (n *Numeric[T]) Get() T {
	return n.value
}

func (n *Numeric[T]) FromString(str string) error {
	var value T
	var err error

	switch any(value).(type) {
	case int:
		parsedValue, parseErr := strconv.Atoi(str)
		if parseErr != nil {
			return parseErr
		}
		value = T(parsedValue)
	case int32:
		parsedValue, parseErr := strconv.ParseInt(str, 10, 32)
		if parseErr != nil {
			return parseErr
		}
		value = T(int32(parsedValue))
	case int64:
		parsedValue, parseErr := strconv.ParseInt(str, 10, 64)
		if parseErr != nil {
			return parseErr
		}
		value = T(parsedValue)
	case float32:
		parsedValue, parseErr := strconv.ParseFloat(str, 32)
		if parseErr != nil {
			return parseErr
		}
		value = T(float32(parsedValue))
	case float64:
		parsedValue, parseErr := strconv.ParseFloat(str, 64)
		if parseErr != nil {
			return parseErr
		}
		value = T(parsedValue)
	case uint:
		parsedValue, parseErr := strconv.ParseUint(str, 10, 0)
		if parseErr != nil {
			return parseErr
		}
		value = T(uint(parsedValue))
	case uint32:
		parsedValue, parseErr := strconv.ParseUint(str, 10, 32)
		if parseErr != nil {
			return parseErr
		}
		value = T(uint32(parsedValue))
	case uint64:
		parsedValue, parseErr := strconv.ParseUint(str, 10, 64)
		if parseErr != nil {
			return parseErr
		}
		value = T(parsedValue)
	default:
		return fmt.Errorf("unsupported type")
	}
	n.Set(value)
	return err
}

func lessThan[T NumberConstraint](a, b T) bool {
	return a < b
}

func greaterThan[T NumberConstraint](a, b T) bool {
	return a > b
}

func equal[T NumberConstraint](a, b T) bool {
	return a == b
}

func (n *Numeric[T]) SetMinValue(min T) {
	n.minValue = &min
}

func (n *Numeric[T]) GetMinValue() T {
	return *n.minValue
}

func (n *Numeric[T]) SetMaxValue(max T) {
	n.maxValue = &max
}

func (n *Numeric[T]) GetMaxValue() T {
	return *n.maxValue
}

func (n *Numeric[T]) SetIncrementStep(step T) {
	n.incrementStep = step
}

func (n *Numeric[T]) SetCompareCondition(f func(T, T) int) {
	n.compareCondition = f
}

func (n *Numeric[T]) AddThreshold(value T, onCross func(T, T)) *Numeric[T] {
	n.thresholds = append(n.thresholds, Threshold[T]{value: value, onCross: onCross})
	return n
}

func (n *Numeric[T]) Apply(f func(T) T) T {
	newValue := f(n.value)
	n.Set(newValue)
	return n.value
}

func (n *Numeric[T]) Validate(validateFunc func(T) bool) bool {
	return validateFunc(n.value)
}

func (n *Numeric[T]) Reset() {
	n.minValue = nil
	n.maxValue = nil
	n.onUpdate = nil
	n.onIncrease = nil
	n.onDecrease = nil
	n.thresholds = nil
}

func (n *Numeric[T]) ResetLimits() {
	n.minValue = nil
	n.maxValue = nil
}

func (n *Numeric[T]) ClearOnUpdate() {
	n.onUpdate = nil
}

func (n *Numeric[T]) ClearOnIncrease() {
	n.onIncrease = nil
}

func (n *Numeric[T]) ClearOnDecrease() {
	n.onDecrease = nil
}

func (n *Numeric[T]) Clone() *Numeric[T] {
	return &Numeric[T]{
		value:      n.value,
		minValue:   n.minValue,
		maxValue:   n.maxValue,
		onUpdate:   n.onUpdate,
		onIncrease: n.onIncrease,
		onDecrease: n.onDecrease,
		thresholds: append([]Threshold[T]{}, n.thresholds...),
		format:     n.format,
	}
}

func (n *Numeric[T]) FormatAsScientific() string {
	return fmt.Sprintf("%e", n.value)
}

func (n *Numeric[T]) FormatAsFixed() string {
	return fmt.Sprintf("%f", n.value)
}

// #region Formating

func (n *Numeric[T]) Serialize() []byte {
	return []byte(fmt.Sprintf("%v", n.value))
}

func (n *Numeric[T]) Deserialize(data []byte) error {
	var value T
	var err error

	switch any(value).(type) {
	case int:
		parsedValue, parseErr := strconv.Atoi(string(data))
		if parseErr != nil {
			return parseErr
		}
		value = T(parsedValue)
	case int32:
		parsedValue, parseErr := strconv.ParseInt(string(data), 10, 32)
		if parseErr != nil {
			return parseErr
		}
		value = T(int32(parsedValue))
	case int64:
		parsedValue, parseErr := strconv.ParseInt(string(data), 10, 64)
		if parseErr != nil {
			return parseErr
		}
		value = T(parsedValue)
	case float32:
		parsedValue, parseErr := strconv.ParseFloat(string(data), 32)
		if parseErr != nil {
			return parseErr
		}
		value = T(float32(parsedValue))
	case float64:
		parsedValue, parseErr := strconv.ParseFloat(string(data), 64)
		if parseErr != nil {
			return parseErr
		}
		value = T(parsedValue)
	case uint:
		parsedValue, parseErr := strconv.ParseUint(string(data), 10, 0)
		if parseErr != nil {
			return parseErr
		}
		value = T(uint(parsedValue))
	case uint32:
		parsedValue, parseErr := strconv.ParseUint(string(data), 10, 32)
		if parseErr != nil {
			return parseErr
		}
		value = T(uint32(parsedValue))
	case uint64:
		parsedValue, parseErr := strconv.ParseUint(string(data), 10, 64)
		if parseErr != nil {
			return parseErr
		}
		value = T(parsedValue)
	default:
		return fmt.Errorf("unsupported type")
	}

	n.Set(value)
	return err
}

func (n Numeric[T]) String() string {
	if n.format == "" {
		return fmt.Sprintf("%v", n.value)
	}
	return fmt.Sprintf(n.format, n.value)
}

func (n *Numeric[T]) Ptr() *T {
	return &n.value
}

func (n *Numeric[T]) FormatAsHex() string {
	switch any(n.value).(type) {
	case int, int32, int64, uint, uint32, uint64:
		return fmt.Sprintf("0x%x", n.value)
	default:
		return "Not applicable for this type"
	}
}

func (n *Numeric[T]) FormatAsBinary() string {
	switch any(n.value).(type) {
	case int, int32, int64, uint, uint32, uint64:
		return fmt.Sprintf("0b%b", n.value)
	default:
		return "Not applicable for this type"
	}
}

func (n *Numeric[T]) NotifyThresholds(oldValue, newValue T) {
	for _, threshold := range n.thresholds {
		if (lessThan(oldValue, threshold.value) && !lessThan(newValue, threshold.value)) ||
			(greaterThan(oldValue, threshold.value) && !greaterThan(newValue, threshold.value)) {
			if threshold.onCross != nil {
				threshold.onCross(newValue, threshold.value)
			}
		}
	}
}

//#endregion

// #region Math operations

func (n *Numeric[T]) Sum(value T) *Numeric[T] {
	n.Set(n.value + value)
	return n
}

func (n *Numeric[T]) Subtract(value T) *Numeric[T] {
	n.Set(n.value - value)
	return n
}

func (n *Numeric[T]) Multiply(value T) *Numeric[T] {
	n.Set(n.value * value)
	return n
}

func (n *Numeric[T]) Divide(value T) *Numeric[T] {
	var err error
	result := T(0)
	if equal(value, T(0)) {
		err = fmt.Errorf("division by zero is not allowed")
	} else {
		result = n.value / value
	}
	n.Set(result, err)
	return n
}

func (n *Numeric[T]) Difference(value T) T {
	return n.value - value
}

func (n *Numeric[T]) Abs() T {
	if lessThan(n.value, T(0)) {
		n.Set(T(0) - n.value)
	}
	return n.value
}

func (n *Numeric[T]) Increment() *Numeric[T] {
	return n.Sum(T(1))
}

func (n *Numeric[T]) Decrement() *Numeric[T] {
	return n.Subtract(T(1))
}

func (n *Numeric[T]) Random(min, max T) *Numeric[T] {
	rand.Seed(time.Now().UnixNano())
	switch any(min).(type) {
	case int, int32, int64, uint, uint32, uint64:
		randomValue := T(rand.Intn(int(max-min+1))) + min
		n.Set(randomValue)
		return n
	case float32, float64:
		randomValue := T(rand.Float64()*float64(max-min) + float64(min))
		n.Set(randomValue)
		return n
	default:
		return n
	}
}

func (n *Numeric[T]) Mean(values ...T) float64 {
	sum := n.value
	count := 1
	for _, v := range values {
		sum = sum + v
		count++
	}
	return float64(sum) / float64(count)
}

func (n *Numeric[T]) Median(values ...T) float64 {
	all := append(values, n.value)
	sort.Slice(all, func(i, j int) bool { return lessThan(all[i], all[j]) })
	length := len(all)
	if length%2 == 0 {
		midLeft := all[length/2-1]
		midRight := all[length/2]
		sum := midLeft + midRight
		return float64(sum) / 2.0
	}
	return float64(all[length/2])
}

func (n *Numeric[T]) Mode(values ...T) []T {
	all := append(values, n.value)
	frequency := make(map[T]int)
	for _, v := range all {
		frequency[v]++
	}
	maxFreq := 0
	for _, freq := range frequency {
		if freq > maxFreq {
			maxFreq = freq
		}
	}
	modes := []T{}
	for v, freq := range frequency {
		if freq == maxFreq {
			modes = append(modes, v)
		}
	}
	return modes
}

func (n *Numeric[T]) Max(values ...T) T {
	max := n.value
	for _, v := range values {
		if greaterThan(v, max) {
			max = v
		}
	}
	return max
}

func (n *Numeric[T]) Min(values ...T) T {
	min := n.value
	for _, v := range values {
		if lessThan(v, min) {
			min = v
		}
	}
	return min
}

func (n *Numeric[T]) Power(exp int) *Numeric[T] {
	result := T(1)
	base := n.value
	for exp > 0 {
		if exp%2 == 1 {
			result = result * base
		}
		base = base * base
		exp /= 2
	}
	n.Set(result)
	return n
}

func (n *Numeric[T]) Root(exp float64) (*Numeric[T], error) {
	var err error
	result := T(0)
	if lessThan(n.value, T(0)) {
		err = fmt.Errorf("cannot calculate root of negative number")
	} else if exp <= 0 {
		err = fmt.Errorf("degree of root must be positive")
	} else {
		result = T(math.Pow(float64(n.value), 1.0/float64(exp)))
	}

	n.Set(result, err)
	return n, nil
}

func (n *Numeric[T]) Variance(values ...T) float64 {
	mean := n.Mean(values...)
	sumOfSquares := 0.0
	for _, v := range append(values, n.value) {
		sumOfSquares += math.Pow(float64(v)-mean, 2)
	}
	return sumOfSquares / float64(len(values)+1)
}

func (n *Numeric[T]) StandardDeviation(values ...T) float64 {
	return math.Sqrt(n.Variance(values...))
}

func (n *Numeric[T]) ApplyOperations(operations ...func(T) T) *Numeric[T] {
	newValue := n.value
	for _, op := range operations {
		newValue = op(newValue)
	}
	n.Set(newValue)
	return n
}

func (n *Numeric[T]) Round() T {
	switch v := any(n.value).(type) {
	case float32:
		return T(float32(math.Round(float64(v))))
	case float64:
		return T(math.Round(v))
	default:
		return n.value
	}
}

func (n *Numeric[T]) RoundTo(places int) T {
	switch v := any(n.value).(type) {
	case float32:
		factor := float32(math.Pow(10, float64(places)))
		return T(float32(math.Round(float64(v)*float64(factor)) / float64(factor)))
	case float64:
		factor := math.Pow(10, float64(places))
		return T(math.Round(v*factor) / factor)
	default:
		return n.value
	}
}

func (n *Numeric[T]) Ceil() T {
	switch v := any(n.value).(type) {
	case float32:
		return T(float32(math.Ceil(float64(v))))
	case float64:
		return T(math.Ceil(v))
	default:
		return n.value
	}
}

func (n *Numeric[T]) Floor() T {
	switch v := any(n.value).(type) {
	case float32:
		return T(float32(math.Floor(float64(v))))
	case float64:
		return T(math.Floor(v))
	default:
		return n.value
	}
}

func (n *Numeric[T]) Quotient(value T) (T, error) {
	if equal(value, T(0)) {
		return n.value, fmt.Errorf("division by zero is not allowed")
	}

	return n.value, fmt.Errorf("unsupported type for Quotient")
}

func (n *Numeric[T]) Remainder(value T) (T, error) {
	if equal(value, T(0)) {
		return n.value, fmt.Errorf("division by zero is not allowed")
	}
	if _, ok := any(n.value).(int); ok {
		return T(int(n.value) % int(value)), nil
	}
	if _, ok := any(n.value).(int32); ok {
		return T(int32(n.value) % int32(value)), nil
	}
	if _, ok := any(n.value).(int64); ok {
		return T(int64(n.value) % int64(value)), nil
	}
	if _, ok := any(n.value).(uint); ok {
		return T(uint(n.value) % uint(value)), nil
	}
	if _, ok := any(n.value).(uint32); ok {
		return T(uint32(n.value) % uint32(value)), nil
	}
	if _, ok := any(n.value).(uint64); ok {
		return T(uint64(n.value) % uint64(value)), nil
	}
	return n.value, fmt.Errorf("unsupported type for Remainder")
}

func (n *Numeric[T]) Factorial() (T, error) {
	if n.value < T(0) {
		return n.value, fmt.Errorf("factorial of negative number is not defined")
	}
	result := T(1)
	for i := T(1); i <= n.value; i = i + T(1) {
		result = result * i
	}
	return result, nil
}

func (n *Numeric[T]) Log(base float64) (T, error) {
	if base <= 1.0 {
		return n.value, fmt.Errorf("logarithm base must be greater than 1")
	}
	if _, ok := any(n.value).(float32); ok {
		return T(math.Log(float64(n.value)) / math.Log(base)), nil
	}
	if _, ok := any(n.value).(float64); ok {
		return T(math.Log(float64(n.value)) / math.Log(base)), nil
	}
	return n.value, fmt.Errorf("logarithm is not defined for this type")
}

func (n *Numeric[T]) Exp() T {
	switch v := any(n.value).(type) {
	case float32:
		return T(float32(math.Exp(float64(v))))
	case float64:
		return T(math.Exp(v))
	default:
		return n.value
	}
}

func (n *Numeric[T]) Log10() (T, error) {
	switch v := any(n.value).(type) {
	case float32:
		return T(math.Log10(float64(v))), nil
	case float64:
		return T(math.Log10(v)), nil
	default:
		return n.value, fmt.Errorf("logarithm base-10 is not defined for this type")
	}
}

func (n *Numeric[T]) Square() *Numeric[T] {
	return n.Multiply(n.value)
}

func (n *Numeric[T]) Sqrt() (T, error) {
	if lessThan(n.value, T(0)) {
		return T(0), fmt.Errorf("cannot calculate square root of a negative number")
	}
	switch v := any(n.value).(type) {
	case float32:
		return T(float32(math.Sqrt(float64(v)))), nil
	case float64:
		return T(math.Sqrt(v)), nil
	default:
		return n.value, fmt.Errorf("square root is not defined for this type")
	}
}

//#endregion

// #region Logical operations
func (n *Numeric[T]) CompareTo(other *Numeric[T]) int {
	if n.compareCondition != nil {
		return n.compareCondition(n.value, other.value)
	}

	if n.value > other.value {
		return 1
	} else if n.value < other.value {
		return -1
	}
	return 0
}

func (n *Numeric[T]) IsPositive() bool {
	return n.value > 0
}

func (n *Numeric[T]) IsNegative() bool {
	return n.value < 0
}

func (n *Numeric[T]) IsEven() bool {
	switch any(n.value).(type) {
	case int, int32, int64, uint, uint32, uint64:
		return int(n.value)%2 == 0
	default:
		return false
	}
}

func (n *Numeric[T]) IsOdd() bool {
	switch any(n.value).(type) {
	case int, int32, int64, uint, uint32, uint64:
		return int(n.value)%2 != 0
	default:
		return false
	}
}

func (n *Numeric[T]) IsInRange(min, max T) bool {
	return n.value >= min && n.value <= max
}

func (n *Numeric[T]) IsMultipleOf(divisor T) bool {
	if divisor == 0 {
		return false
	}

	switch any(n.value).(type) {
	case int, int32, int64, uint, uint32, uint64:
		return int(n.value)%int(divisor) == 0
	case float32, float64:
		return float64(n.value) == 0 || float64(n.value) == float64(int64(n.value)/int64(divisor))*float64(divisor)
	default:
		return false
	}
}

func (n *Numeric[T]) IsEqual(value T) bool {
	return n.value == value
}

func (n *Numeric[T]) IsInSequence(seq []T) bool {
	for _, v := range seq {
		if n.value == v {
			return true
		}
	}
	return false
}

func (n *Numeric[T]) IsPrime() bool {
	switch any(n.value).(type) {
	case int:
		val := int(n.value)
		if val <= 1 {
			return false
		}
		for d := 2; d*d <= val; d++ {
			if val%d == 0 {
				return false
			}
		}
		return true
	default:
		return false
	}
}

//#endregion
//#region Callbacks

func (i *Numeric[T]) OnUpdate(f func(T)) {
	i.onUpdate = f
}

func (i *Numeric[T]) OnIncrease(f func(T, T)) {
	i.onIncrease = f
}

func (n *Numeric[T]) OnDecrease(f func(T, T)) *Numeric[T] {
	n.onDecrease = f
	return n
}

func (n *Numeric[T]) OnOperation(f func(string, T)) *Numeric[T] {
	n.onOperation = f
	return n
}

func (n *Numeric[T]) OnError(f func(error)) *Numeric[T] {
	n.onError = f
	return n
}

func (n *Numeric[T]) OnOverflow(callback func(oldValue, newValue T)) *Numeric[T] {
	n.onOverflow = callback
	return n
}

func (n *Numeric[T]) OnUnderflow(callback func(oldValue, newValue T)) *Numeric[T] {
	n.onUnderflow = callback
	return n
}

//#endregion
