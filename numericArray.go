package mkparam

import (
	"fmt"
	"reflect"
	"sort"
)

type NumericArray[T NumberConstraint] struct {
	elements []*Numeric[T]
	cache    map[string]interface{}

	// Callbacks
	onAdd    func(*Numeric[T], int)
	onRemove func(*Numeric[T], int)
	onUpdate func(T, T, int)
	onClear  func()
}

// NewNumericArrayWithUnique создает массив с уникальными значениями.
func NewNumericArrayWithUnique[T NumberConstraint](values []T) *NumericArray[T] {
	unique := make(map[T]struct{})
	na := &NumericArray[T]{
		cache: make(map[string]interface{}),
	}

	for _, value := range values {
		if _, found := unique[value]; !found {
			unique[value] = struct{}{}
			na.Add(&Numeric[T]{value: value})
		}
	}
	return na
}

// AddValues добавляет значения разных типов в массив Numeric.
func (na *NumericArray[T]) AddValues(values ...interface{}) error {
	na.ClearCache() // Очищаем кеш, так как массив изменился.
	for _, value := range values {
		var numeric *Numeric[T]
		switch v := value.(type) {
		case int:
			numeric = &Numeric[T]{value: T(v)}
		case int32:
			numeric = &Numeric[T]{value: T(v)}
		case int64:
			numeric = &Numeric[T]{value: T(v)}
		case float32:
			numeric = &Numeric[T]{value: T(v)}
		case float64:
			numeric = &Numeric[T]{value: T(v)}
		case uint:
			numeric = &Numeric[T]{value: T(v)}
		case uint32:
			numeric = &Numeric[T]{value: T(v)}
		case uint64:
			numeric = &Numeric[T]{value: T(v)}
		default:
			return fmt.Errorf("unsupported type: %s", reflect.TypeOf(value))
		}
		na.Add(numeric)
	}
	return nil
}

// AddMany добавляет несколько объектов Numeric в массив.
func (na *NumericArray[T]) Add(nums ...*Numeric[T]) {
	na.ClearCache() // Очищаем кеш, так как массив изменился.
	for _, num := range nums {
		na.elements = append(na.elements, num)
		if na.onAdd != nil {
			na.onAdd(num, len(na.elements)-1)
		}
	}
}

// AddArray добавляет элементы из другого массива NumericArray в текущий массив.
func (na *NumericArray[T]) AddArray(other *NumericArray[T]) {
	na.ClearCache() // Очищаем кеш, так как массив изменился.
	startIndex := len(na.elements)
	na.elements = append(na.elements, other.elements...)

	// Вызываем событие OnAdd для каждого нового элемента.
	if na.onAdd != nil {
		for i, elem := range other.elements {
			na.onAdd(elem, startIndex+i)
		}
	}
}

func (na *NumericArray[T]) Concat(arrays ...*NumericArray[T]) {
	na.ClearCache() // Очищаем кеш, так как массив изменился.
	for _, array := range arrays {
		na.elements = append(na.elements, array.elements...)
	}
}

// Remove удаляет объект Numeric по индексу.
func (na *NumericArray[T]) Remove(index int) error {
	if index < 0 || index >= len(na.elements) {
		return fmt.Errorf("index out of range")
	}
	na.ClearCache() // Очищаем кеш, так как массив изменился.
	removedElement := na.elements[index]
	na.elements = append(na.elements[:index], na.elements[index+1:]...)
	if na.onRemove != nil {
		na.onRemove(removedElement, index)
	}
	return nil
}

func (na *NumericArray[T]) Update(index int, newValue T) error {
	if index < 0 || index >= len(na.elements) {
		return fmt.Errorf("index out of range")
	}
	oldValue := na.elements[index].Get()
	na.elements[index].value = newValue
	na.ClearCache() // Очищаем кеш, так как массив изменился.
	if na.onUpdate != nil {
		na.onUpdate(oldValue, newValue, index)
	}
	return nil
}

// Get возвращает объект Numeric по индексу.
func (na *NumericArray[T]) Get(index int) (*Numeric[T], error) {
	if index < 0 || index >= len(na.elements) {
		return nil, fmt.Errorf("index out of range")
	}
	return na.elements[index], nil
}

//#region operations

func (na *NumericArray[T]) Clear() {
	na.elements = nil
	na.ClearCache() // Очищаем кеш, так как массив изменился.
	if na.onClear != nil {
		na.onClear()
	}
}

func (na *NumericArray[T]) Find(value T) (*Numeric[T], bool) {
	for _, elem := range na.elements {
		if elem.Get() == value {
			return elem, true
		}
	}
	return nil, false
}

func (na *NumericArray[T]) Copy() *NumericArray[T] {
	copyArray := &NumericArray[T]{cache: make(map[string]interface{})}
	copyArray.elements = make([]*Numeric[T], len(na.elements))
	copy(copyArray.elements, na.elements)
	return copyArray
}

func (na *NumericArray[T]) Values() []T {
	values := make([]T, len(na.elements))
	for i, elem := range na.elements {
		values[i] = elem.Get()
	}
	return values
}

func (na *NumericArray[T]) Distinct() *NumericArray[T] {
	unique := make(map[T]struct{})
	distinctArray := &NumericArray[T]{cache: make(map[string]interface{})}

	for _, elem := range na.elements {
		if _, found := unique[elem.Get()]; !found {
			unique[elem.Get()] = struct{}{}
			distinctArray.Add(elem)
		}
	}
	return distinctArray
}

func (na *NumericArray[T]) Reverse() {
	na.ClearCache() // Очищаем кеш, так как массив изменился.
	for i, j := 0, len(na.elements)-1; i < j; i, j = i+1, j-1 {
		na.elements[i], na.elements[j] = na.elements[j], na.elements[i]
	}
}

// Map применяет функцию f к каждому элементу массива и возвращает новый массив с результатами.
func (na *NumericArray[T]) Map(f func(T) T) *NumericArray[T] {
	mappedArray := &NumericArray[T]{cache: make(map[string]interface{})}
	for _, elem := range na.elements {
		mappedArray.Add(&Numeric[T]{value: f(elem.Get())})
	}
	return mappedArray
}

// Batch разбивает массив на подмассивы фиксированного размера.
func (na *NumericArray[T]) Batch(size int) []*NumericArray[T] {
	if size <= 0 {
		return nil
	}

	var batches []*NumericArray[T]
	for i := 0; i < len(na.elements); i += size {
		end := i + size
		if end > len(na.elements) {
			end = len(na.elements)
		}
		batch := &NumericArray[T]{cache: make(map[string]interface{})}
		batch.elements = na.elements[i:end]
		batches = append(batches, batch)
	}
	return batches
}

func (na *NumericArray[T]) GroupBy(keyFunc func(T) int) map[int]*NumericArray[T] {
	groups := make(map[int]*NumericArray[T])

	for _, elem := range na.elements {
		key := keyFunc(elem.Get())
		if groups[key] == nil {
			groups[key] = &NumericArray[T]{cache: make(map[string]interface{})}
		}
		groups[key].Add(elem)
	}

	return groups
}

//#endregion

// #region filter
func (na *NumericArray[T]) FindAllIndices(predicate func(T) bool) []int {
	indices := []int{}
	for i, elem := range na.elements {
		if predicate(elem.Get()) {
			indices = append(indices, i)
		}
	}
	return indices
}

func (na *NumericArray[T]) Filter(predicate func(T) bool) []*Numeric[T] {
	filtered := []*Numeric[T]{}
	for _, elem := range na.elements {
		if predicate(elem.Get()) {
			filtered = append(filtered, elem)
		}
	}
	return filtered
}

func (na *NumericArray[T]) Sort(comparator func(a, b T) bool) {
	na.ClearCache() // Очищаем кеш, так как массив изменился.
	sort.Slice(na.elements, func(i, j int) bool {
		return comparator(na.elements[i].Get(), na.elements[j].Get())
	})
}

func (na *NumericArray[T]) FindIndex(value T) int {
	for i, elem := range na.elements {
		if elem.Get() == value {
			return i
		}
	}
	return -1
}

//#endregion
//#region math

func (na *NumericArray[T]) Sum() T {
	if cached, found := na.GetCachedResult("sum"); found {
		return cached.(T)
	}
	var sum T
	for _, num := range na.elements {
		sum += num.Get()
	}
	na.SetCachedResult("sum", sum)
	return sum
}

// Reduce применяет функцию f к элементам массива, сводя их к одному значению.
func (na *NumericArray[T]) Reduce(initial T, f func(T, T) T) T {
	result := initial
	for _, elem := range na.elements {
		result = f(result, elem.Get())
	}
	return result
}

// Count возвращает количество элементов в массиве.
func (na *NumericArray[T]) Count() int {
	if cached, found := na.GetCachedResult("count"); found {
		return cached.(int)
	}
	count := len(na.elements)
	na.SetCachedResult("count", count)
	return count
}

func (na *NumericArray[T]) Min() (T, error) {
	if cached, found := na.GetCachedResult("min"); found {
		return cached.(T), nil
	}
	if len(na.elements) == 0 {
		var zero T
		return zero, fmt.Errorf("array is empty")
	}

	min := na.elements[0].Get()
	for _, elem := range na.elements {
		if elem.Get() < min {
			min = elem.Get()
		}
	}
	na.SetCachedResult("min", min)
	return min, nil
}

func (na *NumericArray[T]) Max() (T, error) {
	if cached, found := na.GetCachedResult("max"); found {
		return cached.(T), nil
	}
	if len(na.elements) == 0 {
		var zero T
		return zero, fmt.Errorf("array is empty")
	}

	max := na.elements[0].Get()
	for _, elem := range na.elements {
		if elem.Get() > max {
			max = elem.Get()
		}
	}
	na.SetCachedResult("max", max)
	return max, nil
}

func (na *NumericArray[T]) Average() (T, error) {
	if cached, found := na.GetCachedResult("average"); found {
		return cached.(T), nil
	}
	if len(na.elements) == 0 {
		var zero T
		return zero, fmt.Errorf("array is empty")
	}

	sum := na.Sum()
	average := sum / T(len(na.elements))
	na.SetCachedResult("average", average)
	return average, nil
}

//#endregion

// #region logic

func (na *NumericArray[T]) IsEmpty() bool {
	return len(na.elements) == 0
}

func (na *NumericArray[T]) Contains(value T) bool {
	for _, elem := range na.elements {
		if elem.Get() == value {
			return true
		}
	}
	return false
}

func (na *NumericArray[T]) Equals(other *NumericArray[T]) bool {
	if len(na.elements) != len(other.elements) {
		return false
	}

	for i, elem := range na.elements {
		if elem.Get() != other.elements[i].Get() {
			return false
		}
	}

	return true
}

func (na *NumericArray[T]) Any(predicate func(T) bool) bool {
	for _, elem := range na.elements {
		if predicate(elem.Get()) {
			return true
		}
	}
	return false
}

func (na *NumericArray[T]) All(predicate func(T) bool) bool {
	for _, elem := range na.elements {
		if !predicate(elem.Get()) {
			return false
		}
	}
	return true
}

//#endregion

//#region Callbacks

func (na *NumericArray[T]) OnAdd(callback func(*Numeric[T], int)) {
	na.onAdd = callback
}

func (na *NumericArray[T]) OnRemove(callback func(*Numeric[T], int)) {
	na.onRemove = callback
}

func (na *NumericArray[T]) OnUpdate(callback func(T, T, int)) {
	na.onUpdate = callback
}

func (na *NumericArray[T]) OnClear(callback func()) {
	na.onClear = callback
}

//#endregion

// #region cache

// GetCachedResult возвращает кешированный результат, если он существует.
func (na *NumericArray[T]) GetCachedResult(key string) (interface{}, bool) {
	result, found := na.cache[key]
	return result, found
}

// SetCachedResult сохраняет результат в кеш.
func (na *NumericArray[T]) SetCachedResult(key string, value interface{}) {
	na.cache[key] = value
}

// ClearCache очищает весь кеш.
func (na *NumericArray[T]) ClearCache() {
	na.cache = make(map[string]interface{})
}

//#endregion
