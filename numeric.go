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

// Numeric представляет числовое значение с различными функциями и настройками.
type Numeric[T NumberConstraint] struct {
	value            T
	format           string         // Format string for custom string representation.
	compareCondition func(T, T) int // Custom comparison function.
	minValue         *T             // Minimum allowable value (if any).
	maxValue         *T             // Maximum allowable value (if any).
	incrementStep    T

	// Callbacks for various events.
	onUpdate    func(T)         // Called when the value is updated.
	onIncrease  func(T, T)      // Called when the value increases.
	onDecrease  func(T, T)      // Called when the value decreases.
	onOperation func(string, T) // Called when an operation is performed.
	onError     func(error)     // Called when an error occurs.
	onOverflow  func(oldValue, newValue T)
	onUnderflow func(oldValue, newValue T)

	thresholds []Threshold[T] // List of thresholds to notify on crossing.
}

// Set updates the value, applies constraints, triggers callbacks, and notifies about threshold crossings.
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
		value = *n.minValue

		if n.onUnderflow != nil {
			n.onUnderflow(value, *n.minValue)
		}
	}

	// Apply maximum value constraint if defined.
	if n.maxValue != nil && greaterThan(value, *n.maxValue) {
		value = *n.maxValue

		if n.onOverflow != nil {
			n.onOverflow(value, *n.maxValue)
		}
	}

	// Apply maximum value constraint if defined.
	if n.maxValue != nil && greaterThan(value, *n.maxValue) {
		// Trigger the onOverflow callback if defined.

		value = *n.maxValue
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

// / Get returns the current value of the Numeric.
func (n *Numeric[T]) Get() T {
	return n.value
}

// FromString parses the provided string into a numeric value and sets it.
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

// lessThan and greaterThan functions for comparing numeric values.
func lessThan[T NumberConstraint](a, b T) bool {
	return a < b
}

func greaterThan[T NumberConstraint](a, b T) bool {
	return a > b
}

func equal[T NumberConstraint](a, b T) bool {
	return a == b
}

// SetMinValue sets a minimum allowable value for the Numeric.
func (n *Numeric[T]) SetMinValue(min T) {
	n.minValue = &min
}

// SetMaxValue sets a maximum allowable value for the Numeric.
func (n *Numeric[T]) SetMaxValue(max T) {
	n.maxValue = &max
}

func (n *Numeric[T]) SetIncrementStep(step T) {
	n.incrementStep = step
}

// SetCompareCondition sets a custom comparison function for comparing the Numeric's value with another value.
func (n *Numeric[T]) SetCompareCondition(f func(T, T) int) {
	n.compareCondition = f
}

// AddThreshold adds a threshold to the Numeric's list of thresholds.
func (n *Numeric[T]) AddThreshold(value T, onCross func(T, T)) *Numeric[T] {
	// Directly add the threshold to the list without wrapping the onCross function
	n.thresholds = append(n.thresholds, Threshold[T]{value: value, onCross: onCross})
	return n
}

// Apply applies a function to the current value and sets the result as the new value.
func (n *Numeric[T]) Apply(f func(T) T) T {
	newValue := f(n.value)
	n.Set(newValue)
	return n.value
}

// Validate checks if the current value satisfies the provided validation function.
func (n *Numeric[T]) Validate(validateFunc func(T) bool) bool {
	return validateFunc(n.value)
}

// Reset clears all configurations of the Numeric, including constraints, callbacks, and thresholds.
func (n *Numeric[T]) Reset() {
	n.minValue = nil
	n.maxValue = nil
	n.onUpdate = nil
	n.onIncrease = nil
	n.onDecrease = nil
	n.thresholds = nil
}

// ResetLimits clears only the minimum and maximum value constraints.
func (n *Numeric[T]) ResetLimits() {
	n.minValue = nil
	n.maxValue = nil
}

// ClearOnUpdate removes the callback function that is triggered when the value is updated.
func (n *Numeric[T]) ClearOnUpdate() {
	n.onUpdate = nil
}

// ClearOnIncrease removes the callback function that is triggered when the value increases.
func (n *Numeric[T]) ClearOnIncrease() {
	n.onIncrease = nil
}

// ClearOnDecrease removes the callback function that is triggered when the value decreases.
func (n *Numeric[T]) ClearOnDecrease() {
	n.onDecrease = nil
}

// Clone creates and returns a new Numeric instance that is a copy of the current Numeric.
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

// FormatAsScientific returns a string representation of the Numeric value in scientific notation format.
func (n *Numeric[T]) FormatAsScientific() string {
	return fmt.Sprintf("%e", n.value)
}

// FormatAsFixed returns a string representation of the Numeric value in fixed-point notation format.
func (n *Numeric[T]) FormatAsFixed() string {
	return fmt.Sprintf("%f", n.value)
}

// #region Formating

// Serialize converts the current value of the Numeric to a byte slice in decimal format.
// Returns the serialized byte slice representation of the Numeric value.
func (n *Numeric[T]) Serialize() []byte {
	return []byte(fmt.Sprintf("%v", n.value))
}

// Deserialize converts a byte slice representation of a numeric value to a Numeric value.
// Takes a byte slice 'data' as input, which is expected to represent a numeric value in decimal format.
// Returns an error if the byte slice cannot be converted to a numeric value.
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

// String returns a string representation of the Numeric.
// If the 'format' string is empty, it returns a default format: "<value>".
// Otherwise, it uses the provided format string to format the value.
func (n Numeric[T]) String() string {
	if n.format == "" {
		return fmt.Sprintf("%v", n.value)
	}
	return fmt.Sprintf(n.format, n.value)
}

// Ptr returns a pointer to the current Numeric value.
// This allows access to the Numeric value directly as a pointer.
func (n *Numeric[T]) Ptr() *T {
	return &n.value
}

// FormatAsHex returns a string representation of the Numeric value in hexadecimal format.
// The format will be "0x<hex_value>", where <hex_value> is the hexadecimal representation of the Numeric value.
func (n *Numeric[T]) FormatAsHex() string {
	switch any(n.value).(type) {
	case int, int32, int64, uint, uint32, uint64:
		return fmt.Sprintf("0x%x", n.value)
	default:
		return "Not applicable for this type"
	}
}

// FormatAsBinary returns a string representation of the Numeric value in binary format.
// The format will be "0b<binary_value>", where <binary_value> is the binary representation of the Numeric value.
func (n *Numeric[T]) FormatAsBinary() string {
	switch any(n.value).(type) {
	case int, int32, int64, uint, uint32, uint64:
		return fmt.Sprintf("0b%b", n.value)
	default:
		return "Not applicable for this type"
	}
}

// NotifyThresholds checks if the value has crossed any of the defined thresholds.
// It compares the old and new values with the thresholds.
// If a threshold has been crossed, the associated callback function is invoked.
// Takes 'oldValue' and 'newValue' as input parameters, representing the value before and after the change.
// Calls the threshold's onCross callback function if a threshold is crossed.
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
// CompareTo compares the current Numeric value to another Numeric value.
// Takes a pointer to another Numeric 'other' for comparison.
// If a custom comparison function is set, it uses that function to compare values.
// Otherwise, it compares values directly:
// Returns 1 if the current value is greater than 'other.value',
// -1 if the current value is less than 'other.value',
// and 0 if they are equal.
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

// IsPositive checks if the current Numeric value is greater than zero.
// Returns true if the current value is positive, otherwise returns false.
func (n *Numeric[T]) IsPositive() bool {
	return n.value > 0
}

// IsNegative checks if the current Numeric value is less than zero.
// Returns true if the current value is negative, otherwise returns false.
func (n *Numeric[T]) IsNegative() bool {
	return n.value < 0
}

// IsEven checks if the current Numeric value is an even number.
// Returns true if the current value is even (divisible by 2 without remainder),
// otherwise returns false.
func (n *Numeric[T]) IsEven() bool {
	switch any(n.value).(type) {
	case int, int32, int64, uint, uint32, uint64:
		return int(n.value)%2 == 0
	default:
		return false
	}
}

// IsOdd checks if the current Numeric value is an odd number.
// Returns true if the current value is odd (not divisible by 2 without remainder),
// otherwise returns false.
func (n *Numeric[T]) IsOdd() bool {
	switch any(n.value).(type) {
	case int, int32, int64, uint, uint32, uint64:
		return int(n.value)%2 != 0
	default:
		return false
	}
}

// IsInRange checks if the current Numeric value is within the specified range [min, max].
// Takes two Numeric values 'min' and 'max' defining the inclusive range.
// Returns true if the current value is greater than or equal to 'min' and less than or equal to 'max',
// otherwise returns false.
func (n *Numeric[T]) IsInRange(min, max T) bool {
	return n.value >= min && n.value <= max
}

// IsMultipleOf checks if the current Numeric value is a multiple of the given Numeric value 'divisor'.
// Takes a Numeric value 'divisor' to check divisibility.
// Returns false if 'divisor' is zero, as division by zero is not allowed.
// Returns true if the current value is divisible by 'divisor' (n.value % divisor == 0),
// otherwise returns false.
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

// IsEqual checks if the current Numeric value is equal to the given Numeric value 'value'.
// Takes a Numeric value 'value' for comparison.
// Returns true if the current value is equal to 'value', otherwise returns false.
func (n *Numeric[T]) IsEqual(value T) bool {
	return n.value == value
}

// IsInSequence checks if the current Numeric value is present in the provided sequence of Numeric values.
// Takes a slice of Numeric values 'seq' to check for inclusion.
// Returns true if the current value is found in 'seq', otherwise returns false.
func (n *Numeric[T]) IsInSequence(seq []T) bool {
	for _, v := range seq {
		if n.value == v {
			return true
		}
	}
	return false
}

// IsPrime checks if the current Numeric value is a prime number.
// A prime number is greater than 1 and divisible only by 1 and itself.
// Returns true if the current value is a prime number, otherwise returns false.
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

// OnUpdate sets a callback function to be called when the Int value is updated.
// Takes a function 'f' that receives the new value of the Int.
// This function will be called whenever the Int value is updated.
func (i *Numeric[T]) OnUpdate(f func(T)) {
	i.onUpdate = f
}

// OnIncrease sets a callback function to be called when the Int value is increased.
// Takes a function 'f' that receives the new value and the previous value of the Int.
// This function will be called whenever the Int value is increased.
func (i *Numeric[T]) OnIncrease(f func(T, T)) {
	i.onIncrease = f
}

// OnDecrease sets a callback function to be called when the Int value is decreased.
// Takes a function 'f' that receives the new value and the previous value of the Int.
// This function will be called whenever the Int value is decreased.
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
